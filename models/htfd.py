from concurrent.futures import ProcessPoolExecutor
import copy
import logging
import os

import cupy
import cupyx.scipy.special
import numpy
import scipy.linalg
import scipy.sparse.linalg
from numba import jit, prange

from models.base_separator import (BaseSeparator, UnknownNdarrayModuleError,
                                   check_naninf, logger)
from models.cupy_sparse_ext.sparse_gtsv import batched_gtsv

# utility funcs
accumulate_local_X = cupy.ElementwiseKernel(
    'T Xcomp, S l, S n_frames, S n_harmonics, S width, S n_freqs', 'raw T X',
    '''
    // n_bases x n_frames x n_harmonics x width
    // i = k*n_frames*n_harmonics*width + t*n_harmonics*width + n*width + w
    const int t = (i/width/n_harmonics)%n_frames;
    assert(t>=0 && t<=n_frames-1);
    if(l>=0 && l<n_freqs){
        int X_index[]={l,t};
        atomicAdd(&X[X_index], Xcomp);
    }
    ''', 'accumulate_local_X_v2')

compute_Ylambda_helper = cupy.ElementwiseKernel(
    'T Xcomp, S l, raw T X, raw T Y, S n_frames, S n_harmonics, S width, S n_freqs',
    'T Ylambda', '''
    // i = k*n_frames*n_harmonics*width + t*n_harmonics*width + n*width + w
    const int t = (i/width/n_harmonics)%n_frames;
    assert(t>=0 && t<=n_frames-1);
    if(l>=0 && l<n_freqs){
        int X_index[]={l,t};
        Ylambda = (Xcomp/X[X_index])*Y[X_index];
    }else{
        Ylambda = 0.0;
    }
    ''', 'compute_Ylambda_helper_v2')

compute_G_helper = cupy.ElementwiseKernel(
    'raw T x, int32 l, raw T ln_omega, T sigma, int32 width, int32 n_freqs',
    'T out', '''
    int k = i/width;
    if(l>=0 && l<n_freqs){
        out = exp(
            -(x[l] - ln_omega[k])*(x[l] - ln_omega[k])/(2.0*sigma*sigma)
        );
    }else{
        out = 0.0;
    }
    ''', 'compute_G_helper_v2')

update_Omega_helper = cupy.ElementwiseKernel(
    'raw T x, int32 l, raw T ln_n, int32 n_harmonics, int32 width, int32 n_freqs',
    'T out', '''
    // n_filters x n_bases x n_frames x n_harmonics x width
    int n_index = (i/width)%n_harmonics;
    if(l>=0 && l<n_freqs){
        out = x[l] - ln_n[n_index];
    }else{
        out = 0.0;
    }
    ''', 'update_Omega_helper_v2')

@jit(nopython=True, parallel=True)
def update_Omega_helper_cpu(x, l_indexes, ln_n, n_harmonics, width, n_freqs):
    n_filters = l_indexes.shape[0]
    n_bases = l_indexes.shape[1]
    n_frames = l_indexes.shape[2]
    out = numpy.zeros((n_filters, n_bases, n_frames, n_harmonics, width), dtype=numpy.float32)
    for f in prange(n_filters):
        for k in prange(n_bases):
            for m in prange(n_frames):
                for n in prange(n_harmonics):
                    for d in prange(width):
                        l = l_indexes[f,k,m,n,d]
                        if l >= 0 and l < n_freqs:
                            out[f,k,m,n,d] = x[l] - ln_n[n]
    return out

def _batched_gtsv_cpu(dl, d, du, b):
    return scipy.sparse.linalg.lsqr(scipy.sparse.diags(dl, offsets=-1) + scipy.sparse.diags(d) + scipy.sparse.diags(du, offsets=1), b)[0]

def batched_gtsv_cpu(dl, d, du, B):
    n_bases = dl.shape[0]
    with ProcessPoolExecutor() as executor:
        results = [executor.submit(_batched_gtsv_cpu, dl[k,:], d[k,:], du[k,:], B[k,:]) for k in range(n_bases)]
        for k, r in enumerate(results):
            B[k,:] = r.result()

def update_A_helper_cpu(omega, powerspec, n_DAP_iter, A, pole_mag_thresh, eps, new_A):
    mask = omega < numpy.pi
    with ProcessPoolExecutor() as executor:
        results = [executor.submit(DAP, omega[f][mask[f]], powerspec[f][mask[f]], n_DAP_iter, A[f,:], pole_mag_thresh, eps) for f in range(omega.shape[0])]
        for f, r in enumerate(results):
            new_A[f,:] = r.result()


@jit(nopython=True, parallel=True)
def compute_Ylambda_helper_cpu(Xcomp, l_indexes, X, Y, n_frames, n_harmonics, width, n_freqs, Ylambda):
    n_bases = Xcomp.shape[0]
    for m in prange(n_frames):
        for k in prange(n_bases):
            for n in prange(n_harmonics):
                for d in prange(width):
                    l = l_indexes[k, m, n, d]
                    if l >= 0 and l < n_freqs:
                        Ylambda[k, m, n, d] = (Xcomp[k, m, n, d]/X[l,m])*Y[l,m]
                    else:
                        Ylambda[k, m, n, d] = 0.0

@jit(nopython=True, parallel=True)
def accumulate_local_X_cpu(Xcomp, l_indexes, n_frames, n_harmonics, width, n_freqs, X):
    '''

    Args:
        Xcomp (numpy.ndarray[float]): n_bases x n_frames x n_harmonics x width
        l_indexes (numpy.ndarray[int]): n_bases x n_frames x n_harmonics x width
    '''
    n_bases = Xcomp.shape[0]
    for m in prange(n_frames):
        for k in range(n_bases):
            for n in range(n_harmonics):
                for d in range(width):
                    l = l_indexes[k, m, n, d]
                    if l >= 0 and l < n_freqs:
                        X[l, m] += Xcomp[k, m, n, d]

@jit(nopython=True, parallel=True)
def compute_G_cpu(x, l_indexes, ln_omega, sigma, width, n_freqs, G):
    '''

    Args:
        x (numpy.ndarray[float]): n_freqs
        l_indexes (numpy.ndarray[int]): n_bases x n_frames x n_harmonics x width
        ln_omege (numpy.ndarray[float]): n_bases x n_frames x n_harmonics
    '''
    n_bases = l_indexes.shape[0]
    n_frames = l_indexes.shape[1]
    n_harmonics = l_indexes.shape[2]
    for m in prange(n_frames):
        for k in range(n_bases):
            for n in range(n_harmonics):
                for d in range(width):
                    l = l_indexes[k, m, n, d]
                    if l >= 0 and l < n_freqs:
                        G[k, m, n, d] = numpy.exp(
                            -(x[l] - ln_omega[k, m, n])**2 /
                            (2.0 * sigma * sigma))
                    else:
                        G[k, m, n, d] = 0.0

def DAP(omega: numpy.ndarray, values: numpy.ndarray, n_iter: int, a: numpy.ndarray, pole_mag_thresh: float = 0.99, eps: float=1e-10):
    '''Optimize discrete all-pole model

    Args:
        omega (xp.ndarray): Angular frequencies (n_freqbins (n_freqs*n_harmonics))
        values (xp.ndarray): Power spectrum values (n_freqbins (n_freqs*n_harmonics))
        n_iter (int): # of iterations
        a (xp.ndarray): Initial values of LPC (filter_deg+1)
        pole_mag_thresh (float): Maximum values of the magnitudes of the poles

    Returns:
        xp.ndarray: LPC (filter_deg+1)
    '''
    xp = cupy.get_array_module(a)
    if omega.ndim > 1 or values.ndim > 1 or a.ndim > 1:
        raise ValueError('omega, values, and a should be 1-dim.')
    if omega.shape[0] != values.shape[0]:
        raise ValueError('omega and values should have the same shape.')

    n_freqbins = omega.shape[0]
    filter_deg = a.shape[0] - 1

    omega_prod_p = (omega[:, None] * xp.arange(filter_deg + 1)[None, :]) # n_freqbins x P+1
    C = xp.cos(omega_prod_p)
    cos_omega_prod_p = xp.cos(-omega_prod_p)
    sin_omega_prod_p = xp.sin(-omega_prod_p)

    def compute_freqz():
        denom = (cos_omega_prod_p @ xp.asarray(a))**2 + (sin_omega_prod_p @ xp.asarray(a))**2
        return xp.maximum(eps, 1.0 / denom) # n_freqbins

    V_dash = (xp.maximum(eps, values[None, :]) @ C).ravel()  # P+1
    if xp == cupy:
        V_dash = V_dash.get()
    for iter in range(n_iter):
        inv_squared_Az = compute_freqz()
        V = (inv_squared_Az[None, :] @ C).ravel()  # P+1
        #
        if xp == cupy:
            a = a.get()
            V = V.get()
        try:
            a[:] = scipy.linalg.solve_toeplitz(V_dash, scipy.linalg.toeplitz(V) @ a)
        except:
            raise ValueError(
                os.linesep.join([
                    'Encountered NaN or Inf', f'{str(V)}', f'{str(V_dash)}',
                    f'{str(a)}'
                ]))
        r = numpy.roots(a)
        abs_r = numpy.abs(r).astype('float64')
        if abs_r.max() > pole_mag_thresh:
            r[abs_r > pole_mag_thresh] = (r[abs_r > pole_mag_thresh] / abs_r[abs_r > pole_mag_thresh]) * pole_mag_thresh
            a[:] = a[0] * numpy.poly(r)
        if xp == cupy:
            a = cupy.asarray(a)
    return a

class NakagamiDistribution(object):
    '''Nakagami distribution utilities
    '''
    @staticmethod
    def E_w2(a, b, xp):
        check_naninf(b, xp)
        return b

    @staticmethod
    def E_w(a, b, xp, approx=False):
        if approx:
            # this corresponds to first order approximation
            # Generally, Gamma(x+a)/Gamma(x) \approx x^a.
            # So, Gamma(x+1/2)/Gamma(x)*sqrt(b/a) \approx sqrt(a)*sqrt(b/a) = sqrt(b)
            return xp.sqrt(NakagamiDistribution.E_w2(a, b, xp))
        else:
            if xp == numpy:
                return xp.exp(scipy.special.gammaln(a + 0.5) - scipy.special.gammaln(a)) * xp.sqrt(b / a)
            elif xp == cupy:
                # exact
                return xp.exp(cupyx.scipy.special.gammaln(a + 0.5) - cupyx.scipy.special.gammaln(a)) * xp.sqrt(b / a)
            else:
                raise UnknownNdarrayModuleError(xp)

    @staticmethod
    def E_ln_w(a, b, xp, approx=False):
        if approx:
            return xp.log(NakagamiDistribution.E_w2(a, b, xp)) * 0.5
        else:
            if xp == numpy:
                return (scipy.special.digamma(a) - xp.log(a / b)) / 2.0
            elif xp == cupy:
                return (cupyx.scipy.special.digamma(a) - xp.log(a / b)) / 2.0
            else:
                raise UnknownNdarrayModuleError(xp)

# Separator
class HTFD(BaseSeparator):
    '''Harmonic-temporal factor decomposition (HTFD)
    '''
    def init_priors(self, lnF0s, alpha_U=0.1, sigma=numpy.log(2.0)/60.0, dt=0.01, alphas_Omega=[1.0,1.0]):
        '''Prior initialization

        Args:
            lnF0s (numpy.ndarray): Center log-fundamental frequencies of spectral bases [log rad]
            alpha_U (float): Hyperparameter of U
            sigma (float): \sigma of log-normal wavelet
            dt (float): time delta of frames
            alphas_Omega ([float,float]): Weights of global and local prior distributions for \Omega
        '''
        self.alpha_global = numpy.float32(alphas_Omega[0])
        self.alpha_local = numpy.float32(alphas_Omega[1])
        self.sigma = numpy.float32(sigma)
        self._const = numpy.float32(sigma*numpy.sqrt(2.0*numpy.pi)/self.dx)
        #
        self.alpha_U = numpy.float32(alpha_U)
        self.beta_U = 1.0
        self.mu = lnF0s.astype('f') # n_bases
        self.tau2 = numpy.float32((numpy.log(2) / 12 / (1/6) *dt))**2.0
        self.v2 = numpy.float32((numpy.log(2) / 12.0 / 3.0))**2.0
        self._transferred_arrays += ["mu"]

    def init_params(self, decay=1.0, input_normalization="average"):
        ''' Parameter initialization

        Args:
            decay (float): Exponential decay factor of magnitudes of harmonic partials
            input_normalization (str): Normalization method of input spectrogram
        '''
        self._input_normalization = input_normalization
        self.w = numpy.exp(-decay*numpy.arange(0,self.n_harmonics))
        self.w /= self.w.sum()
        self.w = numpy.tile(self.w[None,:], (self.n_bases,1)).astype('f') # n_bases x n_harmonics
        self.Omega = numpy.tile(self.mu[:,None], (1, self.n_frames)).astype('f') # n_bases x n_frames
        self.U = self.random_state.uniform(0.0, 1.0, size=(self.n_bases, self.n_frames)).astype('f')
        ########
        self._transferred_arrays += ["Omega", "w", "U"]

    def reconstruct(self, k_list=None):
        '''Compute model spectrogram

        Args:
            k_list (list[int]): Used basis numbers, If None, all bases are used.
        
        Return:
            Model spectrogram (# of frequencies x # of frames)
        '''
        xp = cupy.get_array_module(self.x)

        if k_list is None:
            k_list = slice(self.n_bases)

        X = xp.zeros((self.n_freqs, self.n_frames), dtype='f')

        # prepare
        ln_n = xp.log(xp.arange(1, self.n_harmonics+1)).astype(X.dtype) # n_harmonics
        width = int(max(self.min_calc_range, numpy.round(self.calc_width*self.sigma*2.0/self.dx)))
        width_range = xp.arange(width).astype('i') # width

        def compute_lnG():
            lnnF0 = ln_n[None,None,:] + self.Omega[:,:,None] # n_bases x n_frames x n_harmonics
            leftsides = lnnF0 - self.x[0] - width/2.0*self.dx
            leftsides = xp.clip(xp.around(leftsides/self.dx).astype('i'), 0, self.n_freqs-1 - width)
            indexes = leftsides[:,:,:,None] + width_range[None,None,None,:] # n_bases x n_frames x n_harmonics x width
            lnG = -(self.x[indexes] - lnnF0[:,:,:,None])**2 /(2.0*self.sigma**2)
            return lnG, indexes

        def compute_X():
            lnG, indexes = compute_lnG() # n_bases x n_frames x n_harmonics x width
            Xcomp = xp.exp(lnG) * (self.U[:,:,None,None] * self.w[:,None,:,None]) # n_bases x n_frames x n_harmonics x width
            X[:] = 0.0
            Xcomp = xp.ascontiguousarray(Xcomp[k_list,:,:,:]) # n_valid_bases x n_frames x n_harmonics x width
            indexes = xp.ascontiguousarray(indexes[k_list,:,:,:])
            if xp == numpy:
                accumulate_local_X_cpu(Xcomp, indexes, *Xcomp.shape[1:], self.n_freqs, X)
            else:
                accumulate_local_X(Xcomp, indexes, *Xcomp.shape[1:], self.n_freqs, X)

        compute_X()
        return X

    def fit(self, org_Y, n_iter=100, update_flags=[True,True,True], post_process=None, log_interval=10):
        '''Fit model to `org_Y`

        Args:
            org_Y (xp.ndarray): Observed magnitude spectrogram, n_freqs x n_frames
            n_iter (int): # of iterations
            update_flags (list[bool]): Update flags for U, q(w^2), A, and Omega
            post_process (func): Post process ivoked every `log_interval` iterations
            log_interval (int): # of iterations for logging and post process
        '''
        xp = cupy.get_array_module(org_Y)

        # normalization
        Y = self.normalize_spectrogram(org_Y)
        X = xp.zeros_like(Y)

        # prepare
        ln_n = xp.log(xp.arange(1, self.n_harmonics+1)).astype(X.dtype) # n_harmonics
        width = int(max(self.min_calc_range, numpy.round(self.calc_width*self.sigma*2.0/self.dx)))
        width_range = xp.arange(width).astype('i') # width

        Ylambda = xp.zeros((self.n_bases, self.n_frames, self.n_harmonics, width), dtype=X.dtype)

        # U: n_bases x n_frames
        # w: n_bases x n_harmonics
        # Omega: n_bases x n_frames
        def compute_lnG():
            Omega_ln_n = ln_n[None,None,:] + self.Omega[:,:,None] # n_bases x n_frames x n_harmonics
            leftsides = Omega_ln_n - self.x[0] - width/2.0*self.dx
            leftsides = xp.clip(
                xp.around(leftsides/self.dx).astype('i'),
                0, self.n_freqs-1 - width
            )
            indexes = leftsides[:,:,:,None] + width_range[None,None,None,:] # n_bases x n_frames x n_harmonics x width
            lnG = -(self.x[indexes] - Omega_ln_n[:,:,:,None]).astype('f')**2 /(2.0*self.sigma**2)
            return lnG, indexes

        def compute_X():
            lnG, indexes = compute_lnG() # n_bases x n_frames x n_harmonics x width
            Xcomp = xp.exp(lnG) * (self.U[:,:,None,None] * self.w[:,None,:,None]) # n_bases x n_frames x n_harmonics x width
            X[:] = 0.0
            # accumulate_local_X(Xcomp, indexes, *Xcomp.shape[1:], self.n_freqs, X)
            if xp == numpy:
                accumulate_local_X_cpu(Xcomp, indexes, *Xcomp.shape[1:], self.n_freqs, X)
            else:
                accumulate_local_X(Xcomp, indexes, *Xcomp.shape[1:], self.n_freqs, X)

        def update_Ylambda():
            lnG, indexes = compute_lnG() # n_bases x n_frames x n_harmonics x width
            Xcomp = xp.exp(lnG) * (self.U[:,:,None,None] * self.w[:,None,:,None]) # n_bases x n_frames x n_harmonics x width
            X[:] = 0.0
            # accumulate_local_X(Xcomp, indexes, *Xcomp.shape[1:], self.n_freqs, X)
            if xp == numpy:
                accumulate_local_X_cpu(Xcomp, indexes, *Xcomp.shape[1:], self.n_freqs, X)
            else:
                accumulate_local_X(Xcomp, indexes, *Xcomp.shape[1:], self.n_freqs, X)
            # X[:] = xp.maximum(self.eps, X[:])
            Ylambda[:] = 0.0
            if xp == numpy:
                compute_Ylambda_helper_cpu(Xcomp, indexes, X, Y, *Xcomp.shape[1:], self.n_freqs, Ylambda) # n_bases x n_frames x n_harmonics x width                
            else:
                compute_Ylambda_helper(Xcomp, indexes, X, Y, *Xcomp.shape[1:], self.n_freqs, Ylambda) # n_bases x n_frames x n_harmonics x width
            return indexes

        def update_U():
            numU = Ylambda.sum(axis=(2,3)) + self.alpha_U-1
            numU[numU<0]=0
            denomU = self._const + self.beta_U
            newU = xp.maximum(self.eps, numU/denomU)
            return newU

        def update_w():
            new_w = Ylambda.sum(axis=(1,3))/xp.maximum(self.eps, self._const*self.U.sum(axis=1,keepdims=True)) # n_bases x n_harmonics
            new_w = xp.maximum(self.eps, new_w/new_w.sum(axis=1,keepdims=True))
            return new_w

        def update_Omega(l_indexes):
            # prepare
            main_diag_of_DtD = xp.pad(xp.ones((self.n_frames-2,), dtype=self.Omega.dtype)*2, ((1,1),), mode="constant", constant_values=1.0) # n_frames

            # denom
            denom_main_diag = Ylambda.sum(axis=(2,3))/(self.sigma**2) # n_bases x n_frames
            denom_main_diag += self.alpha_global / self.v2 + main_diag_of_DtD[None,:] * (self.alpha_local / self.tau2)
            denom_lower_diag = -xp.ones((self.n_bases, self.n_frames-1), dtype=self.Omega.dtype) * (self.alpha_local / self.tau2)

            # numel
            # TODO: use update_Omega_helper
            new_Omega = (((self.x[l_indexes] - ln_n[None,None,:,None]) * Ylambda).sum(axis=(2,3))/(self.sigma**2)) # n_bases x n_frames
            new_Omega += self.mu[:,None]*self.alpha_global/self.v2

            # solve tridiagonal systems
            if xp == numpy:
                batched_gtsv_cpu(denom_lower_diag, denom_main_diag, denom_lower_diag, new_Omega)
            else:
                batched_gtsv(denom_lower_diag, denom_main_diag, denom_lower_diag, new_Omega)
            return new_Omega

        def get_loss():
            X[:] = xp.maximum(X, self.eps)
            ll = (Y * xp.log(X) - X).sum()
            prior = -((self.Omega - self.mu[:,None])**2/(2*self.v2)).sum() * self.alpha_global
            prior += -((self.Omega[:,:self.n_frames-1] - self.Omega[:,1:])**2/(2*self.tau2)).sum() * self.alpha_local
            # U prior
            prior += ((self.alpha_U-1)*xp.log(self.U) - self.beta_U*self.U).sum()
            return -(ll+prior)

        for iter in range(n_iter):
            if update_flags[0]:
                comp_freq_indexes = update_Ylambda()
                self.U[:] = update_U()
            if update_flags[1]:
                comp_freq_indexes = update_Ylambda()
                self.w[:] = update_w()
            if update_flags[2]:
                comp_freq_indexes = update_Ylambda()
                self.Omega[:] = update_Omega(comp_freq_indexes)
            compute_X()
            # get loss
            if iter == 0 or (iter+1)%log_interval==0 or iter == n_iter - 1:
                loss = get_loss()
                logger.info("{}/{}: {}".format(iter+1, n_iter, loss))
                if post_process is not None:
                    post_process(iter+1)

class SFHTFD(BaseSeparator):
    '''Source-filter harmonic-temporal factor decomposition (SF-HTFD)

    '''
    @property
    def E_w(self):
        return NakagamiDistribution.E_w(self.w_a,
                                        self.w_b,
                                        self.xp,
                                        approx=self.use_approx_Nakagami)

    @property
    def E_ln_w(self):
        return NakagamiDistribution.E_ln_w(self.w_a,
                                           self.w_b,
                                           self.xp,
                                           approx=self.use_approx_Nakagami)

    @property
    def E_w2(self):
        return NakagamiDistribution.E_w2(self.w_a, self.w_b, self.xp)

    @property
    def w_mask(self):
        '''

        Returns:
            xp.ndarray: whether the frequency is out of bound,  n_filters x n_bases x n_frames x n_harmonics
        '''
        xp = self.xp
        n = xp.arange(1, self.n_harmonics + 1).astype(
            self.Omega.dtype)  # n_bases x n_frames
        return xp.exp(
            self.Omega[:, :, :, None]) * n[None, None, None, :] < numpy.pi

    def _compute_squared_Az(self):
        xp = self.xp
        n = xp.arange(1, self.n_harmonics + 1).astype(
            self.Omega.dtype)  # n_bases x n_frames
        omega = xp.exp(self.Omega[:, :, :, None]) * n[
            None, None,
            None, :]  # n_filters x n_bases x n_frames x n_harmonics
        # A: n_filters x filter_deg+1
        real = xp.zeros(
            (self.n_filters, self.n_bases, self.n_frames, self.n_harmonics),
            dtype=omega.dtype)
        imag = xp.zeros(
            (self.n_filters, self.n_bases, self.n_frames, self.n_harmonics),
            dtype=omega.dtype)
        for q in range(self.A.shape[1]):
            real += xp.cos(-omega * q) * self.A[:, q, None, None, None]
            imag += xp.sin(-omega * q) * self.A[:, q, None, None, None]
        squared_Az = real * real + imag * imag
        valid_mask = omega < numpy.pi
        return squared_Az, valid_mask

    @property
    def inv_squared_Az(self):
        squared_Az, valid_mask = self._compute_squared_Az()
        F = 1.0 / squared_Az
        check_naninf(F, self.xp)
        return F, valid_mask

    def init_params(self, n_filters, filter_degree=16, normalize_ar_coeffs=True,input_normalization="average", use_approx_Nakagami=False):
        ''' Parameter initialization

        Args:
            n_filters (int): # of filters
            filter_degree (int): Filter degree
            normalize_ar_coeffs (bool): If True, normalize a so that a <- a/a[0]
            input_normalization (str): Input normalization method
            use_approx_Nakagami (bool): If True, the expectations related to the Nakagami distribution are approximately computed (slightly fast but not exact).
        '''
        self.use_approx_Nakagami = use_approx_Nakagami
        self.n_filters = n_filters
        self._input_normalization = input_normalization
        self.Omega = numpy.tile(self.mu[None, :, None], (self.n_filters, 1, self.n_frames)).astype('f')  # n_filters x n_bases x n_frames
        initA = numpy.poly(numpy.ones(filter_degree - 1) * 0.1 + self.random_state.uniform(-0.01, 0.01, size=(filter_degree - 1, ))).astype('f')
        self.A = numpy.tile(initA[None, :], (self.n_filters, 1))  # n_filters x filter_degree+1
        self.normalize_ar_coeffs = normalize_ar_coeffs
        if self.normalize_ar_coeffs:
            self.A /= self.A[:, 0, None]

        self.w_a = numpy.ones((self.n_filters, self.n_bases, self.n_frames, self.n_harmonics), dtype='f')  # n_filters x n_bases x n_frames x n_harmonics
        inv_squared_Az, valid_mask = self.inv_squared_Az
        self.w_b = (2.0 * inv_squared_Az).astype('f')  # n_filters x n_bases x n_frames x n_harmonics
        # self.w_b[self.xp.logical_not(valid_mask)] = self.xp.nan
        self.U = self.random_state.uniform(0.0, 1.0, size=(self.n_filters, self.n_bases, self.n_frames)).astype('f')
        ########
        self._transferred_arrays += ["Omega", "A", "w_a", "w_b", "U"]

    def init_priors(self, lnF0s, alpha_U=0.1, sigma=numpy.log(2.0) / 60.0, dt=0.01, n_DAP_iter=1, pole_mag_thresh=0.99, alphas_Omega=[1, 1]):
        '''Prior initialization

        Args:
            lnF0s (numpy.ndarray): Center log-fundamental frequencies of spectral bases [log-rad]
        '''
        self.pole_mag_thresh = pole_mag_thresh
        self.n_DAP_iter = n_DAP_iter
        self.alpha_global = numpy.float32(alphas_Omega[0])
        self.alpha_local = numpy.float32(alphas_Omega[1])
        self.sigma = numpy.float32(sigma)
        self._const = numpy.float32(sigma * numpy.sqrt(2.0 * numpy.pi) / self.dx)
        #
        self.alpha_U = numpy.float32(alpha_U)
        self.beta_U = 1.0
        self.mu = lnF0s.astype('f')  # n_bases
        self.tau2 = numpy.float32((numpy.log(2) / 12 / (1 / 6) * dt))**2.0
        self.v2 = numpy.float32((numpy.log(2) / 12.0 / 3.0))**2.0
        self._transferred_arrays += ["mu"]

    def fit(self, org_Y, n_iter=100, update_flags=[True, True, True, True], post_process=None, log_interval=10):
        '''

        Args:
            org_Y (xp.ndarray): Observed magnitude spectrogram, n_freqs x n_frames
            n_iter (int): # of iterations
            update_flags (list[bool]): Update flags for U, q(w^2), A, and Omega
            post_process (func): Post process ivoked every `log_interval` iterations
            log_interval (int): # of iterations for logging and post process
        '''
        xp = cupy.get_array_module(org_Y)

        # normalization
        Y = self.normalize_spectrogram(org_Y)

        # prepare
        X = xp.zeros_like(Y)

        ln_n = xp.log(xp.arange(1, self.n_harmonics + 1)).astype(X.dtype)  # n_harmonics
        width = int(max(self.min_calc_range, numpy.round(self.calc_width * self.sigma * 2.0 / self.dx)))
        width_range = xp.arange(width).astype('i')  # width

        # Auxiliary variable: n_filters x n_bases x n_frames x n_harmonics x width (frequency bin)
        Ylambda = xp.zeros((self.n_filters, self.n_bases, self.n_frames, self.n_harmonics, width), dtype=X.dtype)

        # U: n_filters x n_bases x n_frames
        # A: n_filters x filer_degree+1
        # w_a, w_b: n_filters x n_bases x n_frames x n_harmonics
        # Omega: n_bases x n_frames

        def compute_G():
            ln_omega = ln_n[None, None, None, :] + self.Omega[:, :, :, None]  # n_filters x n_bases x n_frames x n_harmonics
            leftsides = ln_omega - self.x[0] - width / 2.0 * self.dx
            leftsides = xp.around(leftsides / self.dx).astype('i')
            indexes = leftsides[:, :, :, :, None] + width_range[None, None, None, None, :]  # n_filters x n_bases x n_frames x n_harmonics x width
            if xp == numpy:
                G = xp.zeros(indexes.shape, dtype='f')
                compute_G_cpu(self.x.astype('f'), indexes.reshape(-1, *indexes.shape[2:]), ln_omega.reshape(-1, *ln_omega.shape[2:]).astype('f'), self.sigma, width, self.n_freqs, G.reshape(-1, *G.shape[2:]))
            elif xp == cupy:
                G = compute_G_helper(self.x, indexes, ln_omega, self.sigma, width, self.n_freqs)
            else:
                raise UnknownNdarrayModuleError(xp)
            return G, indexes

        def compute_X():
            G, indexes = compute_G(
            )  # n_filters x n_bases x n_frames x n_harmonics x width
            valid_mask = self.w_mask  # n_filters x n_bases x n_frames x n_harmonics
            Xcomp = G * self.U[:, :, :, None,
                               None] * self.E_w[:, :, :, :,
                                                None]  # n_filters x n_bases x n_frames x n_harmonics x width
            Xcomp *= valid_mask[..., None]
            X[:] = 0.0
            # accumulate_local_X(Xcomp.reshape(-1, *Xcomp.shape[2:]), indexes.reshape(-1, *indexes.shape[2:]), *[int(_) for _ in Xcomp.shape[2:]], self.n_freqs, X)
            if xp == numpy:
                accumulate_local_X_cpu(Xcomp.reshape(-1, *Xcomp.shape[2:]), indexes.reshape(-1, *indexes.shape[2:]), *[int(_) for _ in Xcomp.shape[2:]], self.n_freqs, X)
            else:
                accumulate_local_X(Xcomp.reshape(-1, *Xcomp.shape[2:]), indexes.reshape(-1, *indexes.shape[2:]), *[int(_) for _ in Xcomp.shape[2:]], self.n_freqs, X)


        def update_Ylambda():
            G, indexes = compute_G()  # n_filters x n_bases x n_frames x n_harmonics x width
            valid_mask = self.w_mask  # n_filters x n_bases x n_frames x n_harmonics
            Xcomp = G * self.U[:, :, :, None, None] * xp.exp(self.E_ln_w[:, :, :, :, None])  # n_filters x n_bases x n_frames x n_harmonics x width
            Xcomp *= valid_mask[..., None]
            #
            X[:] = 0.0
            # accumulate_local_X(Xcomp.reshape(-1, *Xcomp.shape[2:]), indexes.reshape(-1, *indexes.shape[2:]), *[int(_) for _ in Xcomp.shape[2:]], self.n_freqs, X)
            if xp == numpy:
                accumulate_local_X_cpu(Xcomp.reshape(-1, *Xcomp.shape[2:]), indexes.reshape(-1, *indexes.shape[2:]), *[int(_) for _ in Xcomp.shape[2:]], self.n_freqs, X)
            else:
                accumulate_local_X(Xcomp.reshape(-1, *Xcomp.shape[2:]), indexes.reshape(-1, *indexes.shape[2:]), *[int(_) for _ in Xcomp.shape[2:]], self.n_freqs, X)
            #
            Ylambda[:] = 0.0
            if xp == numpy:
                compute_Ylambda_helper_cpu(
                    Xcomp.reshape(-1, *Xcomp.shape[2:]),
                    indexes.reshape(-1, *indexes.shape[2:]), X, Y,
                    *[int(_) for _ in Xcomp.shape[2:]], self.n_freqs,
                    Ylambda.reshape(-1, *Ylambda.shape[2:])
                )  # n_filters x n_bases x n_frames x n_harmonics x width
            else:
                compute_Ylambda_helper(
                    Xcomp.reshape(-1, *Xcomp.shape[2:]),
                    indexes.reshape(-1, *indexes.shape[2:]), X, Y,
                    *[int(_) for _ in Xcomp.shape[2:]], self.n_freqs,
                    Ylambda.reshape(-1, *Ylambda.shape[2:])
                )  # n_filters x n_bases x n_frames x n_harmonics x width
            return indexes

        def update_U():
            numU = Ylambda.sum(axis=(3, 4)) + self.alpha_U - 1
            numU[numU < 0] = 0
            valid_mask = self.w_mask  # n_filters x n_bases x n_frames x n_harmonics
            denomU = self._const * (self.E_w * valid_mask).sum(axis=3) + self.beta_U
            newU = xp.maximum(self.eps, numU / denomU)
            return newU

        def update_q_w():
            ''' entires outside the frequency range are set as is. '''
            valid_mask = self.w_mask  # n_filters x n_bases x n_frames x n_harmonics
            new_w_a = self.w_a.copy()
            new_w_a[valid_mask] = Ylambda.sum(
                axis=4
            )[valid_mask] * 0.5 + 1.0  # n_filters x n_bases x n_frames x n_harmonics
            #
            new_w_b = self.w_b.copy()
            xi = xp.sqrt(self.E_w2[valid_mask])
            inv_squared_Az, _ = self.inv_squared_Az
            new_w_b[valid_mask] = inv_squared_Az[valid_mask] * (
                Ylambda.sum(axis=4)[valid_mask] +
                2.0) / (self._const *
                        (self.U[:, :, :, None] * valid_mask)[valid_mask] *
                        (inv_squared_Az[valid_mask] / xi) + 1.0)
            new_w_b[valid_mask] = xp.maximum(self.eps, new_w_b[valid_mask])
            # new_w_b = xp.maximum(self.eps, new_w_b)
            return new_w_a, new_w_b

        def update_Omega(l_indexes):
            # prepare
            main_diag_of_DtD = xp.pad(xp.ones((self.n_frames - 2, ), dtype=self.Omega.dtype) * 2, ((1, 1), ),mode="constant",constant_values=1.0)  # n_frames

            # denom
            denom_main_diag = Ylambda.sum(axis=(3, 4)) / (self.sigma**2)  # n_filters x n_bases x n_frames
            denom_main_diag += self.alpha_global / self.v2 + main_diag_of_DtD[None, None, :] * (self.alpha_local / self.tau2)
            denom_lower_diag = -xp.ones((self.n_filters, self.n_bases, self.n_frames - 1), dtype=self.Omega.dtype) * (self.alpha_local / self.tau2)

            # numel
            if xp == numpy:
                x_minus_ln_n = update_Omega_helper_cpu(self.x, l_indexes, ln_n, int(self.n_harmonics), int(width), self.n_freqs)
            else:
                x_minus_ln_n = update_Omega_helper(self.x, l_indexes, ln_n, int(self.n_harmonics), int(width), self.n_freqs)
            new_Omega = (x_minus_ln_n * Ylambda).sum(axis=(3, 4)) / (self.sigma**2)  # n_filters x n_bases x n_frames
            new_Omega += self.mu[None, :, None] * self.alpha_global / self.v2

            # solve tridiagonal systems
            if xp == numpy:
                batched_gtsv_cpu(
                    denom_lower_diag.reshape(self.n_filters * self.n_bases, self.n_frames - 1),
                    denom_main_diag.reshape(self.n_filters * self.n_bases, self.n_frames),
                    denom_lower_diag.reshape(self.n_filters * self.n_bases, self.n_frames - 1),
                    new_Omega.reshape(self.n_filters * self.n_bases, self.n_frames)
                )
            else:
                batched_gtsv(
                    denom_lower_diag.reshape(self.n_filters * self.n_bases, self.n_frames - 1),
                    denom_main_diag.reshape(self.n_filters * self.n_bases, self.n_frames),
                    denom_lower_diag.reshape(self.n_filters * self.n_bases, self.n_frames - 1),
                    new_Omega.reshape(self.n_filters * self.n_bases, self.n_frames)
                )
            return new_Omega

        def update_A():
            newA = xp.zeros_like(self.A)  # n_filters x filter_d
            omega = xp.exp(self.Omega[:, :, :, None] + ln_n[None, None, None, :])  # n_filters x n_bases x n_frames x n_harmonics, radian
            powerspec = self.E_w2 / 2.0  # n_filters x n_bases x n_frames x n_harmonics
            if xp == numpy:
                update_A_helper_cpu(omega, powerspec, self.n_DAP_iter, self.A, self.pole_mag_thresh, self.eps, newA)
            else:
                for f in range(self.n_filters):
                    mask = omega[f] < numpy.pi
                    omega_f = omega[f][mask]
                    powerspec_f = powerspec[f][mask]
                    newA[f, :] = DAP(omega_f, powerspec_f, self.n_DAP_iter, self.A[f, :], pole_mag_thresh=self.pole_mag_thresh, eps=self.eps)
            if self.normalize_ar_coeffs:
                newA /= newA[:, 0, None]
            return newA

        def get_loss():
            X[:] = xp.maximum(X, self.eps)
            ll = (Y * xp.log(X) - X).sum()
            # Omega prior: n_bases x n_frames
            Omega_prior = -((self.Omega - self.mu[None, :, None])**2 / (2 * self.v2)).sum() * self.alpha_global
            Omega_prior += -((self.Omega[:, :, :self.n_frames - 1] - self.Omega[:, :, 1:])**2 / (2 * self.tau2)).sum() * self.alpha_local
            # U prior
            U_prior = xp.sum((self.alpha_U - 1) * xp.log(self.U) -
                             self.beta_U * self.U)
            # w2 prior
            inv_squared_Az, valid_mask = self.inv_squared_Az
            tmp = xp.maximum(self.E_ln_w[valid_mask], -1.0e10).sum()
            tmp2 = -(xp.maximum(xp.log(inv_squared_Az[valid_mask]), -1.0e10)).sum()
            tmp3 = -(xp.maximum(self.eps, self.E_w2[valid_mask]) /
                     xp.maximum(self.eps, inv_squared_Az[valid_mask]) / 2.0).sum()
            w_prior = tmp + tmp2 + tmp3
            #
            check_naninf(X, xp)
            check_naninf(Omega_prior, xp)
            check_naninf(U_prior, xp)
            check_naninf(tmp, xp)
            check_naninf(tmp2, xp)
            check_naninf(tmp3, xp)
            #
            prior = Omega_prior + U_prior + w_prior
            return -(ll + prior)

        for iter in range(n_iter):
            # U, w2, A, Omega
            if update_flags[0]:
                comp_freq_indexes = update_Ylambda()
                self.U[:] = update_U()
                check_naninf(self.U, xp)
            if update_flags[1]:
                comp_freq_indexes = update_Ylambda()
                self.w_a[:], self.w_b[:] = update_q_w()
                check_naninf(self.w_a, xp)
                check_naninf(self.w_b, xp)
            if update_flags[2]:
                comp_freq_indexes = update_Ylambda()
                self.Omega[:] = update_Omega(comp_freq_indexes)
                check_naninf(self.Omega, xp)
            if update_flags[3]:
                comp_freq_indexes = update_Ylambda()
                self.A[:] = update_A()
                check_naninf(self.A, xp)
            compute_X()

            # get loss
            if iter == 0 or (iter + 1) % log_interval == 0 or iter == n_iter - 1:
                loss = get_loss()
                logger.info("{}/{}: {}".format(iter + 1, n_iter, loss))
                if post_process is not None:
                    post_process(iter + 1)

    def reconstruct(self, k_list=None, f_list=None):
        '''Compute model spectrogram

        Args:
            k_list (list[int]): Used pitches numbers. If None, all pitches are used.
            f_list (list[int]): Used filter numbers. If None, all filters are used.
        
        Return:
            Model spectrogram (# of frequencies x # of frames)
        '''
        xp = cupy.get_array_module(self.x)

        if k_list is None:
            k_list = [k for k in range(self.n_bases)]
        if f_list is None:
            f_list = [f for f in range(self.n_filters)]

        X = xp.zeros((self.n_freqs, self.n_frames), dtype='f')

        # prepare
        ln_n = xp.log(xp.arange(1, self.n_harmonics + 1)).astype(X.dtype)  # n_harmonics
        width = int(max(self.min_calc_range, numpy.round(self.calc_width * self.sigma * 2.0 / self.dx)))
        width_range = xp.arange(width).astype('i')  # width

        def compute_G():
            ln_omega = ln_n[None, None, None, :] + self.Omega[f_list, :, :, None][:, k_list, :, :]  # n_filters x n_bases x n_frames x n_harmonics
            leftsides = ln_omega - self.x[0] - width / 2.0 * self.dx
            leftsides = xp.around(leftsides / self.dx).astype('i')
            indexes = leftsides[:, :, :, :, None] + width_range[None, None, None, None, :]  # n_filters x n_bases x n_frames x n_harmonics x width
            if xp == numpy:
                G = xp.zeros(indexes.shape, dtype='f')
                compute_G_cpu(self.x.astype('f'), indexes.reshape(-1, *indexes.shape[2:]), ln_omega.reshape(-1, *ln_omega.shape[2:]).astype('f'), self.sigma, width, self.n_freqs, G.reshape(-1, *G.shape[2:]))
            else:
                G = compute_G_helper(self.x, indexes, ln_omega, self.sigma, width, self.n_freqs)
            return G, indexes

        def compute_X():
            G, indexes = compute_G()  # n_filters x n_bases x n_frames x n_harmonics x width
            valid_mask = self.w_mask  # n_filters x n_bases x n_frames x n_harmonics
            Xcomp = G * self.U[f_list, :, :, None, None][:, k_list, :, :, :] * self.E_w[f_list, :, :, :,None][:,k_list, :, :, :]  # n_filters x n_bases x n_frames x n_harmonics x width
            Xcomp *= valid_mask[f_list, :, :, :, None][:, k_list, :, :, :]
            X[:] = 0.0
            if xp == numpy:
                accumulate_local_X_cpu(Xcomp.reshape(-1, *Xcomp.shape[2:]), indexes.reshape(-1, *indexes.shape[2:]), *[int(_) for _ in Xcomp.shape[2:]], self.n_freqs, X)
            else:
                accumulate_local_X(Xcomp.reshape(-1, *Xcomp.shape[2:]), indexes.reshape(-1, *indexes.shape[2:]), *[int(_) for _ in Xcomp.shape[2:]], self.n_freqs, X)

        compute_X()
        return X


