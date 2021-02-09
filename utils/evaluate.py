import numpy
from oct2py import octave
import os

from utils import load_audio

octave.addpath("bss_eval")
octave.addpath("utils")

def compute_bsseval_v2(refs: numpy.ndarray, srcs: numpy.ndarray):
    '''Compute BSSEval metrics by Octave

    Args:
        refs (numpy.ndarray): references (# of srcs x signal length)
        srcs (numpy.ndarray): references (# of srcs x signal length)

    Return:
        numpy.ndarray: BSSEval metrics (# of srcs x 3 (SDR, SIR, SAR))
    '''
    results = octave.compute_bsseval_v2(srcs, refs)
    return results

def prepare_evaluate(conf: dict):
    '''Prepare for evaluation

    Args:
        conf (dict): Configuration of a mixture
    
    Returns:
        tuple[list,numpy.ndarray,numpy.ndarray]: Groundtruth pitches, input SDRs, and groundtruth signals (# of pitches x signal length)
    '''
    gt_list = conf["gt"]
    gt_pitches = sorted([int(p) for p in gt_list.keys()])
    refs = [load_audio(gt_list[p]) for p in gt_pitches]
    refs = numpy.stack(refs, axis=0) # n_pitches x sig_len
    # load mixed
    mixed = load_audio(conf["mix"])
    # compute input sisdr
    sdrs = compute_bsseval_v2(refs, numpy.tile(mixed[None, :refs.shape[1]] / refs.shape[0], (refs.shape[0], 1)))
    return gt_pitches, sdrs, refs

def evaluate_srcs(attrib: tuple,
                  attrib_titles: tuple,
                  separated_signals: dict,
                  gt_pitches: list,
                  input_sisdrs: numpy.ndarray,
                  refs: numpy.ndarray) -> str:
    '''Evalute separated results

    Args:
        attrib (tuple[str]): Attribute list
        separated_signals (dict): Separated signals (# of sources x signal length)
        gt_pitches (list): Groundtruth pitches
        input_sisdrs (numpy.ndarray): Input SI-SDRs (a.k.a. SDRs obtained with BSSEval v2)
        refs (numpy.ndarray): References (# of sources x signal length)
    
    Return:
        str: Result text in the csv format
    '''
    srcs = [separated_signals[p] for p in gt_pitches]
    srcs = numpy.stack(srcs, axis=0)
    src_sdrs = compute_bsseval_v2(refs, srcs)
    improvements = src_sdrs.copy()
    improvements[:, :2] -= input_sisdrs[:, :2]  # n_pitches x 3
    #
    result_text = ",".join(attrib_titles) + ",pitch,SDR_imp,SIR_imp,SAR,input_SDR,input_SIR,input_SAR" + os.linesep
    for i, p in enumerate(gt_pitches):
        result_text += ",".join([str(_) for _ in attrib]) + f",{p}"
        for j in range(improvements.shape[1]):
            result_text += f",{improvements[i,j]}"
        for j in range(input_sisdrs.shape[1]):
            result_text += f",{input_sisdrs[i,j]}"
        result_text += os.linesep
    return result_text

