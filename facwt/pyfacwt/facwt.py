# -*- coding:utf-8 -*-

from .libfacwt import *
import numpy as np


def ndarray2XVec(x):
    """Conversion from numpy.ndarray to c++ vector

    Parameters
    ----------
    x : numpy.ndarray
      numpy array object

    Returns
    -------
    y : IVec and FVec
      C++ vector version of 'x'
    """
    if isinstance(x, np.ndarray):
        if x.ndim == 1:
            if x.dtype == np.float or x.dtype == np.float32:
                y = FVec()
            elif x.dtype == np.int or x.dtype == np.int32:
                y = IVec()
            else:
                raise TypeError(
                    "Unsupported type of x.dtype [{}]".format(x.dtype))
            for d in x:
                y.append(d)
        elif x.ndim == 2:
            if x.dtype == np.complex or x.dtype == np.complex64:
                y = CFVec2d()
                element_type = CFVec
            elif x.dtype == np.float or x.dtype == np.float32:
                y = FVec2d()
                element_type = FVec
            else:
                raise TypeError(
                    "Unsupported type of x.dtype [{}]".format(x.dtype))
            for ii in range(x.shape[0]):
                z = element_type()
                for v in x[ii]:
                    z.append(v)
                y.append(z)
        else:
            raise NotImplementedError()
    elif isinstance(x, list):
        if x[0].dtype == np.complex or x[0].dtype == np.complex64:
            y = CFVec2d()
            element_type = CFVec
        elif x[0].dtype == np.float or x[0].dtype == np.float32:
            y = FVec2d()
            element_type = FVec
        else:
            raise TypeError("Unsupported type of x.dtype [{}]".format(x.dtype))
        for ii in range(len(x)):
            z = element_type()
            for v in x[ii]:
                z.append(v)
            y.append(z)
    else:
        raise NotImplementedError()
    return y


def XVec2ndarray(x):
    """Conversion from c++ vector (c++ vector of vectors) into numpy.ndarray (python list of numpy.ndarray, resp.)

    Parameters
    ----------
    x : IVec, FVec, CFVec, FVec2d and CFVec2d
       C++ vector objects

    Returns
    -------
    y : numpy.ndarray and list of numpy.ndarray
       python version of 'x'
    """
    if isinstance(x, IVec) or isinstance(x, FVec) or isinstance(x, CFVec):
        return np.asarray(list(x))
    elif isinstance(x, FVec2d) or isinstance(x, CFVec2d):
        y = [np.asarray(list(x_slice)) for x_slice in x]
        return y
    else:
        raise TypeError(
            "The type of \"x\" must be for IVec, FVec, CFVec, FVec2d and CFVec2d.")


class FACWT(object):
    """Fast approximate continuous wavelet transform (FACWT) class

    Details of the FACWT algorithm is described in [HKameoka2008Patent] and [Tnakamura2014DAFx].

    Attributes
    ----------
    cwt : cppCWT
       C++ CWT instance in libfacwt
    center_angfreqs : FVec
       center normalized angular frequencies
    sig_len: int
       original signal length
    T : int
       extended signal length
    fs : float
       sampling frequency

    References
    ----------

    .. [HKameoka2008Patent] H. Kameoka, T. Tahara, T. Nishimoto, and S. Shigeki, "Signal processing method and device," Nov. 2008, Japan Patent JP2008-281898. (in Japanese)

    .. [TNakamura2014DAFx] T. Nakamura and H. Kameoka, "Fast signal reconstruction from magnitude spectrogram of continuous wavelet transform based on spectrogram consistency," in Proceedings of International Conference on Digital Audio Effects 2014, pp. 129--135, 2014.

    Examples
    --------
    >>> import numpy as np
    >>> fs = 16000
    >>> data = ndarray2XVec(np.random.rand(fs*10))
    >>> facwt = FACWT(fs*10, 27.5, fs/2, fs, 60, 2.0, 0.02, 1.0, True)
    >>> spectrogram = facwt.forward(data) # forward FACWT
    >>> rec = facwt.backward(spectrogram) # inverse FACWT

    """

    def __init__(self, _sigLen: int, lowFreq: float, highFreq: float, fs: int, resol: int, width: float, sd: float, alpha: float, multirate: bool, minWidth: int=256, waveletType: str="log_normal", verbose: int=0):
        """Initialization

        Parameters
        ----------
        _sigLen : int
           signal length

        lowFreq : float
           lowest center frequency of CWT

        highFreq : float
           highest center frequency of CWT

        fs : int
           sampling frequency

        resol : int
           Resolution of an semitone

        width : float
           Ratio of standard deviation for approximating frequency responses of the filterbank. The frequency responses are assumed to be zero outside the range [-width\sigma,width\sigma] around each center frequency.

        sd : float
           Standard deviation of Gaussian shape in the log-frequency domain.

        alpha : float
           If alpha=1 (2), magnitudes (power, resp.) of frequency responses have Gaussian shape.

        multirate : bool
           If true, setup for multirate spectrogram; otherwise setup for spectrogram with common sampling rate between all channels.

        verbose : int
            For debug print
        """
        self.cwt = cppCWT()
        init_cwt(self.cwt, _sigLen, lowFreq, highFreq,
                 fs, resol, width, sd, alpha, multirate, minWidth, waveletType)
        self.wavelet_type = waveletType
        self.fs = fs
        self.t_grid = []
        latest_time = self.cwt.T / float(fs)
        for l in range(len(self.cwt.freq_resps)):
            D = len(self.cwt.freq_resps[l])
            self.t_grid.append(np.arange(0, D) / float(D - 1) * latest_time)
        self.verbose = verbose

    @property
    def center_angfreqs(self):
        return self.cwt.center_angfreqs

    @property
    def T(self):
        return self.cwt.T

    @property
    def sig_len(self):
        return self.cwt.sigLen

    def forward(self, data: np.ndarray):
        """Forward computation

        Parameters
        ----------
        data : :class:`np.ndarray`
           Signal

        Returns
        -------
        spectrogram : list of :class:`np.ndarray`
           Spectrogram of the signal
        """
        # spectrum = CFVec()
        # spectrogram = CFVec2d()
        # self.cwt.FFT_signal(spectrum, data, True)
        # self.cwt.forward(spectrogram, spectrum, True, False)
        # return spectrogram
        spectrogram = []
        cwt_forward(spectrogram, data, self.cwt, self.verbose)
        return spectrogram

    def backward(self, spectrogram: list):
        """Backward computation

        Parameters
        ----------
        spectrogram : list
           Complex spectrogram

        Returns
        -------
        signal : FVec
           Signal of the spectrogram
        """
        # spectrum = CFVec()
        # data = FVec()
        # spectrogram = ndarray2XVec(spectrogram)
        # self.cwt.backward(spectrum, spectrogram, True, False)
        # self.cwt.IFFT_spectrum(data, spectrum, True)
        # data = XVec2ndarray(data)
        data = cwt_backward(spectrogram, self.cwt, self.verbose)
        return data

    def reconstruct_phase(self, spectrogram: CFVec2d, spectrum=None, n_iter: int=100):
        """Phase reconstruction

        Parameters
        ----------
        spectrogram : CFVec2d
           Complex spectrogram

        spectrum : CFVec
           CFVec instance for computation efficiency

        n_iter : int
           The number of iterations

        Returns
        -------
        spectrogram : CFVec2d
           Spectrogram with given magnitudes (magnitudes of 'spectrogram') and reconstrcuted phase, and reconstructed signal

        data : FVec
           Reconstructed signal
        """
        if spectrum is None:
            spectrum = CFVec()
            for _ in range(self.cwt.T // 2 + 1):
                spectrum.append(0.0 + 1j * 0.0)
        if not isinstance(spectrum, CFVec):
            raise TypeError("The type of 'spectrum' must be libfacwt.CFVec.")
        self.cwt.reconstruct_phase(spectrogram, spectrum, n_iter, True)
        data = FVec()
        self.cwt.IFFT_spectrum(data, spectrum, True)
        return spectrogram, data

