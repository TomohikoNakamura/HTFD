import logging
import pathlib

import numpy
import soundfile
import yaml
from pyfacwt import FACWT
from utils import save_audio, load_audio

logger = logging.getLogger('HTFD')

def load_conf(filename: str, return_all_tfr_settings=False):
    '''Load time-frequency transform profile

    Args:
        filename (str): YAML filename
        return_all_tfr_settings (bool): If True, return all settings of time-frequency transforms. Defaults to False.
    
    Return:
        tuple[dict,list]: Time-freuqency transform parameters and separation methods' parameters
    '''
    with open(filename, "r") as fp:
        conf = yaml.load(fp, Loader=yaml.FullLoader)
    tfr_conf = conf["TFR"]
    tfr_set_list = tfr_conf["sets"]
    if return_all_tfr_settings:
        return tfr_set_list, conf["Methods"]
    else:
        logger.info(f'TFR: {tfr_conf["active"]}')
        return tfr_set_list[tfr_conf["active"]], conf["Methods"]


def wav2spectrogram(filename: str,
                    max_length: int = None,
                    sr: int = 16000,
                    start_pos: float=0.0,
                    **kwargs):
    """Convert wavefile to complex CWT spectrogram

    Args:
        filename (str): Wav filename
        max_length (int, optional): Maximum signal length [s]. Defaults to None.
        sr (int, optional): Sampling rate [Hz]. Defaults to 16000.
        start_pos (float, optional): Anlysis start position of waveform [s]. Defaults to 0.0.
        **kwargs: Parameters for pyfacwt.FACWT

    Returns:
        numpy.ndarray: Complex CWT spectrogram
    """    
    wavdata = load_audio(filename, sr=sr, mono=True)
    if start_pos > 0.0:
        wavdata = wavdata[int(start_pos*sr):]
    if max_length is not None:
        wavdata = wavdata[:int(max_length * sr)]
    # setup FACWT
    facwt_params = dict(lowFreq=kwargs.get("lowFreq", 27.5),
                        highFreq=kwargs.get("highFreq", sr / 2),
                        fs=sr,
                        resol=kwargs.get("resol", 3),
                        width=kwargs.get("width", 2.0),
                        sd=kwargs.get("sd", numpy.log(2.0) / 60.0),
                        alpha=kwargs.get("alpha", 1.0),
                        multirate=kwargs.get("multirate", False),
                        minWidth=kwargs.get("minWidth", 2),
                        waveletType=kwargs.get("waveletType", "log_normal"),
                        verbose=kwargs.get("verbose", 1))
    facwt = FACWT(wavdata.shape[0], **facwt_params)

    # forward computation
    spectrogram = facwt.forward(wavdata)
    return facwt, facwt_params, spectrogram


def spectrogram2signal(filename, spectrogram, facwt, save=False, norm=False):
    '''Convert complex CWT spectrogram into a time-domain signal

    Args:
        filename (pathlib.Path): Output filename
        spectrogram (numpy.ndarray): Complex CWT spectrogram
        facwt (FACWT): Fast approximate CWT instance
        save (bool, optional): If true, write the signal to file. Defaults to False.
        norm (bool, optional): If true, normalize the signal. Defaults to False
    
    Return:
        numpy.ndarray: Time-domain signal
    '''
    data = facwt.backward(spectrogram)
    if save:
        data = save_audio(filename, data, facwt.fs, norm=norm)
    return data


