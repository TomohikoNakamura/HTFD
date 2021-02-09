import copy
import os
import argparse
import pathlib
import logging

import cupy
import joblib
import numpy
import yaml
# import pretty_midi
import soundfile
from scipy.interpolate import interp1d
from tqdm import tqdm

import pyfacwt
from utils import (get_F0s, rad_to_midi_note_number, save_audio)
from unsupervised_exp import fit_model
from utils.tfr import load_conf, spectrogram2signal, wav2spectrogram

logger = logging.getLogger("HTFD")

TRANSPOSITION_PROFILES = {
    "to_Minor": {
        "E": "Eb",
        "A": "Ab",
        "B": "Bb"
    },
    "to_Major": {
        "Eb": "E",
        "Ab": "A",
        "Bb": "B"
    }
}

PITCH_CLASSES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

def get_key_transpose_rules(to_major=False, root="C"):
    '''Create pitch class mapping table for a specified scale (major or minor) and root pitch

    Args:
        to_major (bool, optional): If true, convert to major. Defaults to False.
        root (str): Root note. Defaults to "C".

    Return:
        dict[str,int]: Pitch class and required pitch shift (one semitone up or down)
    '''
    rules = {}
    root_index = PITCH_CLASSES.index(root)
    for k, v in TRANSPOSITION_PROFILES["to_Major" if to_major else "to_Minor"].items():
        pitch_class_num = (PITCH_CLASSES.index(k) + root_index) % len(PITCH_CLASSES)
        difference = PITCH_CLASSES.index(v) - PITCH_CLASSES.index(k)
        rules[pitch_class_num] = difference
    return rules


def convert_to_human_friendly_rules(rules: dict) -> dict:
    """Convert rules to human-friendly ones

    Args:
        rules (dict): Pitch transposition rules

    Returns:
        dict: Human-friendly rules
    """
    hf_rule = {}
    for k, difference in rules.items():
        hf_rule[PITCH_CLASSES[k]] = PITCH_CLASSES[(k + difference + len(PITCH_CLASSES)) % len(PITCH_CLASSES)]
    return hf_rule


def convert_musical_key(model_filename: pathlib.Path,
                        org_spectrogram: numpy.ndarray,
                        decimation_factor: int,
                        facwt: pyfacwt.FACWT,
                        facwt_params: dict,
                        F0s: numpy.ndarray,
                        transposition_rules: dict,
                        method: str,
                        method_conf: dict,
                        n_pe_iters: int=400,
                        log_interval: int=100,
                        gpu: int=0,
                        overwrite: bool=False):
    """Convert musical key

    Args:
        model_filename (pathlib.Path): Model filename
        org_spectrogram (numpy.ndarray): Original complex spectrogram
        decimation_factor (int): Decimation factor
        facwt (pyfacwt.FACWT): Fast approximate CWT instance
        facwt_params (dict): FACWT parameters
        F0s (numpy.ndarray): F0s of pitches
        transposition_rules (dict): Pitch transposition rules
        method (str): Separation method
        method_conf (dict): Separation method's parameters
        n_pe_iters (int, optional): # of iterations for phase estimation. Defaults to 400.
        log_interval (int, optional): Log interval of loss. Defaults to 100.
        gpu (int, optional): GPU number. Defaults to 0.
        overwrite (bool, optional): If true, overwrite trained models. Defaults to False.

    Raises:
        NotImplementedError: Pitch transposition error, 'Only one semitone up/down supported.'

    Returns:
        dict[str,numpy.ndarray]: Transposed and separated signals
            "transposed": Transposed signal
            "unchanged": Separated signal of unchanged pitches
            "to-be-transposed": Separated signal of to-be-transposed pitches
    """    
    # Get model parameters and update schedule
    method_params = method_conf.get("params", {})
    param_update_schedule = method_conf["schedule"]

    # Decimate spectrogram so that its frame interval is around 10 ms.
    decimated_spectrogram = org_spectrogram[:, ::decimation_factor].copy()
    # fit model
    fit_model(method, method_params, param_update_schedule, decimated_spectrogram, facwt, facwt_params, F0s, model_filename, log_interval=log_interval, gpu=gpu, overwrite=overwrite)
    model = joblib.load(model_filename)

    # separation
    if gpu >= 0:
        xp = cupy
        model.to_gpu()
    else:
        xp = numpy
        model.to_cpu()
    facwt.verbose = 0  # suppress output

    X = xp.maximum(model.eps, model.reconstruct()).astype('f')
    # X[:] = xp.maximum(model.eps * model.n_bases, X).astype('f')

    def compute_separation_mask(k_list):
        X_k = xp.maximum(model.eps, model.reconstruct(k_list=k_list)).astype('f')
        weight = (X_k / X).astype('f')
        if xp == cupy:
            weight = cupy.asnumpy(weight)

        if decimation_factor > 1:
            interpfun = interp1d(numpy.arange(0, weight.shape[1]) * decimation_factor, weight, kind="linear", axis=1, bounds_error=False, fill_value="extrapolate")
            weight = interpfun(numpy.arange(0, org_spectrogram.shape[1])).astype('f')
            weight[weight < 0] = 0
            weight[weight > 1] = 1

        return weight

    # compute unchanged components
    midi_note_nums = rad_to_midi_note_number(F0s, facwt.fs)
    asis_k_list = []
    to_be_transposed_k_list = []
    for k, midi_note_num in zip(range(model.n_bases), midi_note_nums):
        if midi_note_num % 12 in transposition_rules:
            to_be_transposed_k_list.append(k)
        else:
            asis_k_list.append(k)
    unchanged_mask = compute_separation_mask(asis_k_list)
    transposed_magnitude = numpy.abs(org_spectrogram) * unchanged_mask

    # separated results
    ## unchanged components
    unchanged_complex_spectrogram = org_spectrogram * unchanged_mask
    unchanged_signal = spectrogram2signal(None, list(unchanged_complex_spectrogram), facwt, save=False)

    ## components to be transposed
    to_be_transposed_mask = compute_separation_mask(to_be_transposed_k_list)
    to_be_transposed_complex_spectrogram = org_spectrogram * to_be_transposed_mask
    to_be_transposed_signal = spectrogram2signal(None, list(to_be_transposed_complex_spectrogram), facwt, save=False)

    # add changed components
    logger.info("Transposing pitches")
    transpose_shift_num = facwt_params["resol"]
    changed_magnitude = numpy.zeros_like(transposed_magnitude)
    for k, midi_note_num in zip(range(model.n_bases), midi_note_nums):
        if midi_note_num % 12 in transposition_rules:
            transposed_magnitude_k = numpy.abs(org_spectrogram) * compute_separation_mask([k])
            if transposition_rules[midi_note_num % 12] == -1: # One semitone down
                changed_magnitude[0:-transpose_shift_num, :] += transposed_magnitude_k[transpose_shift_num:, :]
            elif transposition_rules[midi_note_num % 12] == 1: # One semitone up
                changed_magnitude[transpose_shift_num:, :] += transposed_magnitude_k[0:-transpose_shift_num, :]
            else:
                raise NotImplementedError('Only one semitone up/down supported.')
    transposed_magnitude += changed_magnitude

    # phase estimation
    logger.info("Estimating phase")
    initial_phase = numpy.angle(org_spectrogram)
    transposed_spectrogram = (transposed_magnitude * numpy.exp(1j * initial_phase)).astype(org_spectrogram.dtype)
    transposed_spectrogram, transposed_signal = facwt.reconstruct_phase(pyfacwt.ndarray2XVec(transposed_spectrogram), n_iter=n_pe_iters)
    transposed_signal = pyfacwt.XVec2ndarray(transposed_signal)

    return {"transposed": transposed_signal, "unchanged": unchanged_signal, "to-be-transposed": to_be_transposed_signal}


def main():
    parser = argparse.ArgumentParser()
    # IO
    parser.add_argument("-i", "--inputs", nargs="*", required=True, help="Input wav files")
    parser.add_argument("-o", "--output_dir", required=True, help="Output dir")
    parser.add_argument("--keys", choices=[p+" Major" for p in PITCH_CLASSES] + [p+" Minor" for p in PITCH_CLASSES], required=True, help="", nargs="*")
    # configuration
    parser.add_argument("--tf_conf", required=True, help="Config YAML file including parameters of time-frequency transform and methods")
    parser.add_argument("--hopsize", type=float, default=10.0/1000.0, help="Hopsize [s] of decimated CWT spectrograms")
    parser.add_argument("--target_sr", default=16000, type=int, help="Sampling rate [Hz]")
    parser.add_argument("--lowest_freq", type=float, default=27.5, help="Lowest frequency [Hz] of pitches")
    parser.add_argument("--n_semitones", type=int, default=88, help="# of semitones")
    # method
    parser.add_argument("--methods", required=True, nargs="*", help="Methods")
    # signal reconstruction parameters
    parser.add_argument("--n_pe_iters", type=int, default=400, help="# iterations for phase reconstruction")
    # input target
    parser.add_argument("--start_pos", type=float, default=0.0, help="Start position [s]")
    parser.add_argument("--dur", type=float, default=None, help="Duration [s] from the beginning of the musical piece for pitch transposition")
    # output options
    parser.add_argument("--dry_run", action="store_true", help="Only show pitch transposition rules")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite trained models and separated signals")
    parser.add_argument("--normalize_output", action="store_true", help="Normalize transposed waveforms whose maximum magnitude is 0.99.")
    # misc
    parser.add_argument("--gpu", type=int, default=0, help="GPU number (If negative, use only CPU.)")

    # parse
    args = parser.parse_args()

    if len(args.keys) != len(args.inputs):
        parser.error(f'# of keys must equal # of inputs: {len(args.keys)} vs {len(args.inputs)}')


    # Load TFR config
    tf_analysis_param, all_method_conf = load_conf(args.tf_conf)
    tf_analysis_param["start_pos"] = args.start_pos
    tf_analysis_param["minWidth"] = 2
    if args.dur is not None:
        tf_analysis_param["max_length"] = args.dur
        logger.info(f"Wav analysis: {args.dur} [s] from {args.start_pos} [s]")
    else:
        logger.info(f"Wav analysis: From {args.start_pos} [s] to the end")

    # Create output directory
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # wav filename loop
    for target_filename, key_name in zip([pathlib.Path(_) for _ in args.inputs], args.keys):
        logger.info(f"Target: {target_filename}")
        
        # Get transposition rules
        # key_name = get_key_name(target_filename)
        root_pitch_class, major_or_minor = key_name.split()
        transposition_rules = get_key_transpose_rules(to_major="Minor" == major_or_minor, root=root_pitch_class)
        logger.info(transposition_rules)
        logger.info(f"Key transposition rule: {convert_to_human_friendly_rules(transposition_rules)}")

        # Write transposition rules
        trans_conf_filename = output_dir / (target_filename.stem + f"_conf.yaml")
        if not trans_conf_filename.exists():
            with open(trans_conf_filename, "w") as fp:
                converted_key_name = key_name.split(" ")[0] + " natural minor scale" if "Major" in key_name else key_name.split(" ")[0] + " major scale"
                yaml.dump({"file": target_filename, "key": key_name, "converted_key": converted_key_name}, fp)
        
        # If dry run, go to next file
        if args.dry_run:
            continue

        # Perform FA-CWT
        facwt, facwt_params, org_spectrogram = wav2spectrogram(target_filename, sr=args.target_sr, **tf_analysis_param)
        org_spectrogram = numpy.asarray(org_spectrogram) # Convet list of CVecs to numpy.ndarray
        # Get decimation factor
        decimation_factor = max(1, int(numpy.floor(args.hopsize / numpy.diff(facwt.t_grid[0]).mean()))) if args.hopsize > 0 else 1
        logger.info(f"Frame decimation factor: {decimation_factor} sample")

        # Get note F0s
        F0s = get_F0s(facwt.fs, lowest_freq=args.lowest_freq, n_semitones=args.n_semitones)

        # Method loop
        for method in args.methods:
            logger.info(f"Method: {method}")
            basename = target_filename.stem
            output_filename = output_dir / (basename + f"_{method}_transposed.wav")
            unchanged_filename = output_dir / (basename + f"_{method}_unchanged.wav")
            tobetransposed_filename = output_dir / (basename + f"_{method}_tobetransposed.wav")
            model_filename = output_dir / (basename + f"_{method}.jbl")
            #
            if output_filename.exists() and not args.overwrite:
                continue
            # Configure method
            method_conf = copy.deepcopy(all_method_conf[method])
            # Convert key
            result = convert_musical_key(model_filename, org_spectrogram, decimation_factor, facwt, facwt_params, F0s, transposition_rules, method, method_conf, n_pe_iters=args.n_pe_iters, gpu=args.gpu, overwrite=args.overwrite)
            # Save
            logger.info("Save tranposed and separated signals")
            save_audio(output_filename, result["transposed"], sr=facwt.fs, norm=args.normalize_output)
            save_audio(unchanged_filename, result["unchanged"], sr=facwt.fs, norm=args.normalize_output)
            save_audio(tobetransposed_filename, result["to-be-transposed"], sr=facwt.fs, norm=args.normalize_output)

if __name__ == "__main__":
    main()

    
