import copy
import logging

import cupy
import numpy

logger = logging.getLogger('HTFD')

class UnknownNdarrayModuleError(NotImplementedError):
    def __init__(self, xp):
        self.message = f'Unknown ndarray module [{xp}]'

def check_naninf(x, xp):
    '''NaN Inf checker

    Args:
        x (xp.ndarray): Value
        xp (numpy or cupy): ndarray module

    Return:

    '''
    if xp.isinf(x).any():
        raise FloatingPointError
    if xp.isnan(x).any():
        raise FloatingPointError

class BaseSeparator(object):
    '''Separator Base Class

    Attributes:
        n_freqs (int): # of frequency bins
        n_frames (int): # of frames
        n_bases (int): # of bases (normally # of pitches)
        n_harmonics (int): # of harmonics
        eps (float): Machine epsilon
        seed (int): Random seed
        random_state (numpy.random.RandomState): Random state instance
        x (xp.ndarray): Log-center frequencies of CWT [ln rad]
        dx (float): Delta x
        calc_width (float): Computation range factor for Gaussians with center \Omega_{k,m}
        min_calc_range (int): Minimum value of the calculation range.   
        
        _transferred_arrays (list[str]): Variables names transferred to GPU and CPU
        _prior_params (dict): Input dictionary of prior parameters
        _init_params (dict): Input dictionary of initialization parameters
    '''
    def __init__(self, n_frames, n_harmonics, x, lnF0s, seed=0, eps=1e-5, prior_params={}, init_params={}, constant_params={}):
        '''
        
        Args:
            n_frames (int): # of frames
            n_harmonics (int): # of harmonics
            x (numpy.ndarray[float]): log-center frequencies of CWT [ln rad]
            lnF0s (numpy.ndarray[float]): log-fundamental frequencies [ln rad]
            seed (int): random seed
            eps (float): flooring value
            prior_params (dict): args for function init_prior
            init_params (dict): args for function init_param
            constant_params (dict): args for function init_constant_params
        '''
        super().__init__()
        self._transferred_arrays = []
        # initialize parameters
        n_bases, n_freqs = int(lnF0s.shape[0]), int(x.shape[0])  # matrix sizes
        self.n_freqs = n_freqs
        self.n_frames = n_frames
        self.n_bases = n_bases
        self.n_harmonics = n_harmonics
        # misc
        self.eps = eps  # epsilon
        self.seed = seed
        logger.info(f"Random Seed: {seed}")
        self.random_state = numpy.random.RandomState(seed)  # random state
        # constants
        self.x = x.astype('f')
        self._transferred_arrays += ["x"]
        self.dx = numpy.absolute(numpy.diff(self.x)[0])
        self.init_constant_params(**constant_params)
        # initialize prior
        self.init_priors(lnF0s, **prior_params)
        # initialize parameters
        self.init_params(**init_params)
        #
        self._prior_params = copy.deepcopy(prior_params)
        self._init_params = copy.deepcopy(init_params)
        ###################
        logger.info(f"Transferred arrays: {self._transferred_arrays}")

    def init_constant_params(self, calc_width, min_calc_range):
        '''Set constants

        Args:
            calc_width (float): Computation range factor for Gaussians with center \Omega_{k,m}.   
                To accerate the computation of the Gaussians, we compute the elements only within a range of [-\sigma*calc_width,\sigma*calc_width] around each \Omega_{k,m} (the other elements are zeros.)
            min_calc_range (int): Minimum value of the calculation range.   
                For example, if it set to 3, at least the element at x_{l} corresponding to \Omega_{k,m} and its previous and next elements are computed.)
        '''
        self.calc_width = numpy.float32(calc_width)
        self.min_calc_range = int(min_calc_range)

    def init_priors(self, *args, **kwargs):
        raise NotImplementedError('init_priors is not implemented.')

    def init_params(self, *args, **kwargs):
        raise NotImplementedError('init_params is not implemented.')

    def to_gpu(self):
        '''Transfer variables (included in `self._transferred_arrays`) to GPU
        '''
        for name in self._transferred_arrays:
            setattr(self, name, cupy.asarray(getattr(self, name)))

    def to_cpu(self):
        '''Transfer variables (included in `self._transferred_arrays`) to CPU
        '''
        for name in self._transferred_arrays:
            setattr(self, name, cupy.asnumpy(getattr(self, name)))

    @property
    def xp(self):
        '''Get array module

        Return:
            numpy or cupy module
        '''
        return cupy.get_array_module(self.x)

    def normalize_spectrogram(self, org_Y):
        '''Normalize spectrograms

        The normalization method is specified by `self._input_normalization`

        Args:
            org_Y (xp.ndarray[float]): Magnitude/power spectrogram
        
        Return:
            Normalized spectrogram
        '''
        if self._input_normalization == "average":
            Y = org_Y/org_Y.mean()
        elif self._input_normalization == "max":
            Y = org_Y/org_Y.max()
        elif self._input_normalization == "none":
            Y = org_Y.copy()
        else:
            raise NotImplementedError(f"Undefined input normalization method [{self._input_normalization}]")
        return Y
