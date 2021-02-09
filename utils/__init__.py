import logging
import pathlib

import coloredlogs
import librosa
import numpy
import soundfile

# logger
logger = logging.getLogger('HTFD')
logger.setLevel("INFO")
coloredlogs.install(level="INFO", fmt="%(asctime)s [%(levelname).1s] %(message)s")

# imshow parameters for convenience
imshow_params = {
    'aspect': 'auto',
    'interpolation': 'nearest',
    'origin': 'lower'
}

# IO
def load_audio(filename: pathlib.Path, sr: int=None, mono: bool=False):
    '''Load audio

    Args:
        filename (pathlib.Path): Filename
        sr (int, optional): Sampling rate [Hz]: Defaults to None (specified in wav file).
        mono (bool, optional): Make signal monaural. Defaults to False.
    
    Return:
        numpy.ndarray: Signal
    '''
    data, _ = librosa.load(filename, sr=sr, mono=mono)
    return data

def save_audio(filename: pathlib.Path, data: numpy.ndarray, sr: int, norm: bool=False):
    """Save waveform

    Args:
        filename (pathlib.Path): Output filename
        data (numpy.ndarray): Signal
        sr (int): Sampling rate [Hz]
        norm (bool, optional): If true, normalize the signal. Defaults to False

    Returns:
        numpy.ndarray: Signal (If norm=True, normalized one.)
    """
    if norm:
        data /= numpy.max(numpy.abs(data))
        data *= 0.99
    soundfile.write(filename.as_posix(), data, sr, subtype='PCM_16')
    return data

# Naming utility
def get_model_filename(output_dir: str, method: str, input_dir: str, log10_alpha_U: float, filter_degree: int=None, n_filters: int=None) -> str:
    """Get model filename

    Args:
        output_dir (str): Output directory
        method (str): Method
        input_dir (str): Input wav directory
        log10_alpha_U (float): \log_{10} \alpha^{(U)}
        filter_degree (int, optional): Filter degree. Defaults to None.
        n_filters (int, optional): # of filters. Defaults to None.

    Returns:
        str: Model filename
    """
    input_dir = pathlib.Path(input_dir).name
    if filter_degree is None and n_filters is None:
        return pathlib.Path(f"{output_dir}") / f"{method}_log10alphaU{log10_alpha_U:+02.02f}" / input_dir / "model.jbl"
    else:
        return pathlib.Path(f"{output_dir}") / f"{method}_log10alphaU{log10_alpha_U:+02.02f}_deg{filter_degree:02d}_nf{n_filters:02d}" / input_dir / "model.jbl"

# Musical utility
def get_F0s(fs: int, lowest_freq: float=27.5, n_semitones: int=88) -> numpy.ndarray:
    '''Get fundamental frequencies of pitches

    Args:
        fs (int): Sampling frequency
        lowest_freq (float): Frequency of lowest pitch
        n_semitones (int): # of semitones
    
    Return:
        numpy.ndarray: Normalized angular frequencies of pitches [rad]
    '''
    return lowest_freq * (2**(numpy.arange(0.0, n_semitones) / 12)) / fs * (2.0 * numpy.pi)

def rad_to_midi_note_number(F0s: numpy.ndarray, fs: int) -> numpy.ndarray:
    """Convert frequencies [rad] into midi note numbers

    Args:
        F0s (numpy.ndarray): Normalized angular frequencies of pitches [rad]
        fs (int): Sampling frequency

    Returns:
        numpy.ndarray: Midi note numbers
    """    
    return (numpy.log2(F0s / (2.0 * numpy.pi) * fs / 440) * 12 + 69).astype('i')

