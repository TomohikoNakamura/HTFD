# -*- encoding:utf-8 -*-

import copy
import functools
import itertools
import logging
import os
import pathlib
import argparse

import cupy
import joblib
import matplotlib.pyplot as plt
import numpy
import yaml
from scipy.interpolate import interp1d
from tqdm import tqdm

from utils import (get_F0s, imshow_params, load_audio,
                   rad_to_midi_note_number, get_model_filename)
from models import htfd
from utils.tfr import load_conf, spectrogram2signal, wav2spectrogram
import pyfacwt

logger = logging.getLogger('HTFD')

def main(argv=None):
    parser = argparse.ArgumentParser()
    # IO
    parser.add_argument("-i", "--input", required=True, help="Directory including mixture and groundtruths")
    parser.add_argument("-o", "--output_dir", required=True, help="Output directory")
    # configuration
    parser.add_argument("--tf_conf", required=True, help="Config YAML file including parameters of time-frequency transform and methods")
    parser.add_argument("--hopsize", type=float, default=10.0/1000.0, help="Hopsize [s] of decimated CWT spectrograms")
    parser.add_argument("--target_sr", default=16000, type=int, help="Sampling rate [Hz]")
    parser.add_argument("--lowest_freq", type=float, default=27.5, help="Lowest frequency [Hz] of pitches")
    parser.add_argument("--n_semitones", type=int, default=88, help="# of semitones")
    # grid search params
    parser.add_argument("--methods", required=True, nargs="*", help="Methods")
    parser.add_argument("--alpha_Us", type=float, nargs="*", help="\alpha^{(U)} list")
    parser.add_argument("--filter_degrees", type=int, nargs="*", help="Filter degree list")
    parser.add_argument("--n_filters_list", type=int, nargs="*", help="# of filters list")
    # misc
    parser.add_argument("--overwrite", action="store_true", help="If true, overwrite trained models and separated signals")
    parser.add_argument("--save_waveform", action="store_true", help="If true, save separated signals")
    parser.add_argument("--log_interval", default=10, type=int, help="Log interval of loss")
    parser.add_argument("--gpu", type=int, default=0, help="GPU number (If negative, use only CPU.)")
    # parse
    args = parser.parse_args(argv)

    # print args
    logger.info(str(args))

    # Load input directory config
    with open(pathlib.Path(args.input) / "config.yaml", "r") as fp:
        data_conf = yaml.load(fp, Loader=yaml.FullLoader)

    # Load TFR config
    tf_analysis_param, all_method_conf = load_conf(args.tf_conf)

    # Perform CWT
    facwt, facwt_params, spectrogram = wav2spectrogram(data_conf["mix"], sr=args.target_sr, **tf_analysis_param)
    spectrogram = numpy.asarray(spectrogram) # Convet list of CVecs to numpy.ndarray
    # Get decimation factor
    decimation_factor = max(1, int(numpy.floor(args.hopsize / numpy.diff(facwt.t_grid[0]).mean())))
    logger.info(f"frame decimation factor: {decimation_factor} sample")
    # Decimate spectrogram so that its frame interval is around 10 ms.
    decimated_spectrogram = spectrogram[:, ::decimation_factor].copy()

    # Set note F0s and midi note numbers
    F0s = get_F0s(facwt.fs, lowest_freq=args.lowest_freq, n_semitones=args.n_semitones)
    midi_note_nums = rad_to_midi_note_number(F0s, facwt.fs)
    
    # Get required elements for evaluation
    evaluate_args = evaluate_module.prepare_evaluate(data_conf)  # gt_pitches, input_sisdrs, refs

    # Set hyperparameters for grid search
    method_list = args.methods
    log10_alpha_U_list = args.alpha_Us
    filter_degrees_list = args.filter_degrees
    n_filters_list = args.n_filters_list

    # Method loop
    for method in method_list:
        logger.info(f"Start {method}")
        # Copy method conf
        method_conf = copy.deepcopy(all_method_conf[method])
        # Define trial generators for grid search
        if "SFHTFD" in method:
            def trial_generator():
                for log10_alpha_U, filter_degree, n_filters in itertools.product(log10_alpha_U_list, filter_degrees_list, n_filters_list):
                    model_filename = get_model_filename(args.output_dir, method, args.input, log10_alpha_U, filter_degree=filter_degree, n_filters=n_filters)
                    # Set parameters
                    method_params = copy.deepcopy(method_conf.get("params", {}))
                    method_params["prior_params"]["alpha_U"] = 10**log10_alpha_U
                    method_params["init_params"]["filter_degree"] = filter_degree
                    method_params["init_params"]["n_filters"] = n_filters
                    param_update_schedule = method_conf["schedule"]
                    yield model_filename, method_params, param_update_schedule, (args.input, method, log10_alpha_U, filter_degree, n_filters), ("file", "method," "log10_alpha_U", "filter_deg", "n_filters")
        elif "HTFD" in method:
            def trial_generator():
                for log10_alpha_U in log10_alpha_U_list:
                    model_filename = get_model_filename(args.output_dir, method, args.input, log10_alpha_U)
                    # Set parameters
                    method_params = copy.deepcopy(method_conf.get("params", {}))
                    method_params["prior_params"]["alpha_U"] = 10**log10_alpha_U
                    param_update_schedule = method_conf["schedule"]
                    yield model_filename, method_params, param_update_schedule, (args.input, method, log10_alpha_U, -1, -1), ("file", "method," "log10_alpha_U", "filter_deg", "n_filters")
        else:
            raise ValueError(f"Unknown method [{method}]")
        # Do grid search
        for model_filename, method_params, param_update_schedule, attrib, attrib_titles in trial_generator():
            # Fit
            fit_model(method, method_params, param_update_schedule, decimated_spectrogram, facwt, facwt_params, F0s, model_filename, log_interval=args.log_interval, gpu=args.gpu, overwrite=args.overwrite)
            # Evaluate
            summary_filename = model_filename.parent / "summary.txt"
            if not summary_filename.exists() or args.save_waveform:
                # separate
                srcs = separate_signals(method, model_filename, data_conf, midi_note_nums, spectrogram, facwt, facwt_params, decimation_factor, save_waveform=args.save_waveform, gpu=args.gpu, overwrite=args.overwrite)
                # evaluate
                with open(summary_filename, "w") as fp:
                    fp.write(evaluate_module.evaluate_srcs(attrib, attrib_titles, srcs, *evaluate_args))


def fit_model(method: str, method_params: dict, param_update_schedule: dict, spectrogram: numpy.ndarray, facwt: pyfacwt.FACWT, facwt_params: dict, F0s: numpy.ndarray, model_filename: str, log_interval: int=10, gpu: int=-1, overwrite: bool=False):
    """Fit model to observed spectrogram

    Args:
        method (str): Method
        method_params (dict): Parameters of the method
        param_update_schedule (dict): Parameter update schedule
        spectrogram (numpy.ndarray[complex]): Observed spectrogram
        facwt (pyfacwt.FACWT): Fast approximate CWT instance
        facwt_params (dict): FACWT parameters
        F0s (numpy.ndarray): F0s of pitches [rad]
        model_filename (pathlib.Path): Output model filename
        log_interval (int, optional): Log interval of loss. Defaults to 10.

    Raises:
        NotImplementedError: Unknown method
    """
    if not model_filename.exists() or overwrite:
        if "SFHTFD" in method:
            fit_func = functools.partial(fit_SFHTFD, ModelClass=htfd.SFHTFD)
        elif "HTFD" in method:
            fit_func = functools.partial(fit_HTFD, ModelClass=htfd.HTFD)
        else:
            raise NotImplementedError(f"Unknown method [{method}]")
        model = fit_func(facwt, facwt_params, spectrogram, F0s, method_params, param_update_schedule, disp=False, log_interval=log_interval, gpu=gpu)
        model_filename.parent.mkdir(exist_ok=True, parents=True)
        joblib.dump(model, model_filename, compress=True)
        with open(model_filename.parent / "conf.yml", "w") as fp:
            yaml.dump({"params": method_params, "schedule": param_update_schedule}, fp)
    else:
        model = joblib.load(model_filename)

def separate_signals(method: str, model_filename: pathlib.Path, data_conf: dict, midi_note_nums: numpy.ndarray, spectrogram: numpy.ndarray, facwt: pyfacwt.FACWT, facwt_params: dict, decimation_factor: int, save_waveform: bool=False, gpu: int=-1, overwrite: bool=False) -> numpy.ndarray:
    """Separate signals

    Args:
        method (str): Method
        model_filename (pathlib.Path): Model filename
        data_conf (dict): Input data configuration
        midi_note_nums (numpy.ndarray): Midi note numbers of pitches
        spectrogram (numpy.ndarray): Observed complex spectrogram
        facwt (pyfacwt.FACWT): Fast approximate CWT instance
        facwt_params (dict): FACWT parameters
        decimation_factor (int): Decimation factor
        save_waveform (bool, optional): If True, dump separated signals. Defaults to False.
        gpu (int, optional): Gpu number. Defaults to -1.
        overwrite (bool, optional): If True, overwrite separated signals. Defaults to False.

    Returns:
        numpy.ndarray: Separated signals
    """    
    # load model
    model = joblib.load(model_filename)
    # separate preparation
    valid_k_list = list(filter(lambda k: midi_note_nums[k] in data_conf["gt"].keys(), list(range(model.n_bases))))
    logger.info("valid k list: {} {}".format(valid_k_list, data_conf["gt"].keys()))
    if gpu >= 0:
        xp = cupy
        model.to_gpu()
    else:
        xp = numpy
    X = None
    facwt.verbose = 0  # suppress output
    #
    separated_signals = {}
    for k in tqdm(valid_k_list, leave=True, desc='    {0: >10s}'.format('Valid basis ')):
        outfname = model_filename.parent / f"pitch{midi_note_nums[k]:03d}.wav"
        if not outfname.exists() or overwrite:
            if X is None:
                X = model.reconstruct()
                X[:] = xp.maximum(model.eps, X)
                X = X.astype('f')
            X_k = xp.maximum(model.eps, model.reconstruct(k_list=[k])).astype('f')
            weight = (X_k / X).astype('f')

            if xp == cupy:
                weight = cupy.asnumpy(weight)

            # Interpolate masks at decimated frames
            if decimation_factor > 1:
                interpfun = interp1d(numpy.arange(0, weight.shape[1]) * decimation_factor, weight, kind="linear", axis=1, bounds_error=False, fill_value="extrapolate")
                weight = interpfun(numpy.arange(0, spectrogram.shape[1])).astype('f')
                # Squash weights into [0,1]
                weight[weight < 0] = 0
                weight[weight > 1] = 1
            # masking
            Y_k = spectrogram * weight
            # Convert into time-domain signal
            separated_signals[midi_note_nums[k]] = spectrogram2signal(outfname, list(Y_k), facwt, save=save_waveform)
        else:
            separated_signals[midi_note_nums[k]] = load_audio(outfname)[0]
    return separated_signals

def fit_HTFD(facwt: pyfacwt.FACWT, facwt_params: dict, spectrogram: numpy.ndarray, F0s: numpy.ndarray, method_params: dict, param_update_schedule: dict, disp: bool=False, log_interval: int=10, ModelClass: object=None, gpu: int=-1):
    # initialize spectral templates
    x = numpy.log(facwt.center_angfreqs)
    L, M = spectrogram.shape
    N = method_params["n_harmonics"]  # the number of harmonics
    prior_params = dict(**method_params["prior_params"], sigma=facwt_params["sd"], dt=numpy.diff(facwt.t_grid[0]).mean())
    # n_frames, n_harmonics, x, lnF0s
    htfd = ModelClass(n_frames=M, x=x, lnF0s=numpy.log(F0s), **method_params)
    if facwt_params["alpha"] == 2.0:
        logger.info("Use POWER spectrogram")
        target_spectrogram = (numpy.absolute(spectrogram)**2).astype('f')
    elif facwt_params["alpha"] == 1.0:
        logger.info("Use MAGNITUDE spectrogram")
        target_spectrogram = numpy.absolute(spectrogram).astype('f')
    else:
        raise ValueError

    if disp:
        plt.subplot(211)
        plt.imshow(target_spectrogram**0.3, **imshow_params)

        def postproc(X):
            plt.subplot(212)
            plt.imshow(X**0.3, **imshow_params)
            plt.draw()
            plt.pause(1e-3)
    else:
        postproc = None

    if gpu >= 0:
        htfd.to_gpu()
        target_spectrogram = cupy.array(target_spectrogram).astype(cupy.float32)
    logger.info("========== Start training ==========")
    logger.info("Iteration Loss")
    for schedule in param_update_schedule:
        htfd.fit(target_spectrogram,
                 n_iter=schedule["n_iter"],
                 update_flags=schedule["update_flags"],
                 post_process=postproc,
                 log_interval=log_interval)
    logger.info("========== Finish training =========")
    del target_spectrogram
    if gpu >= 0:
        htfd.to_cpu()
    return htfd

def fit_SFHTFD(facwt: pyfacwt.FACWT, facwt_params: dict, spectrogram: numpy.ndarray, F0s: numpy.ndarray, method_params: dict, param_update_schedule: dict, disp: bool=False, log_interval: int=10, ModelClass: object=None, gpu: int=-1):
    # initialize spectral templates
    x = numpy.log(facwt.center_angfreqs)
    L, M = spectrogram.shape
    prior_params = dict(**method_params["prior_params"], sigma=facwt_params["sd"], dt=numpy.diff(facwt.t_grid[0]).mean())
    # n_frames, n_harmonics, x, lnF0s
    htfd = ModelClass(n_frames=M, x=x, lnF0s=numpy.log(F0s), **method_params)
    if facwt_params["alpha"] == 2.0:
        logger.info("Use POWER spectrogram")
        target_spectrogram = (numpy.absolute(spectrogram)**2).astype('f')
    elif facwt_params["alpha"] == 1.0:
        logger.info("Use MAGNITUDE spectrogram")
        target_spectrogram = numpy.absolute(spectrogram).astype('f')
    else:
        raise ValueError

    if disp:
        plt.subplot(211)
        plt.imshow(target_spectrogram**0.3, **imshow_params)

        def postproc(X):
            plt.subplot(212)
            plt.imshow(X**0.3, **imshow_params)
            plt.draw()
            plt.pause(1e-3)
    else:
        postproc = None

    if gpu >= 0:
        htfd.to_gpu()
        target_spectrogram = cupy.array(target_spectrogram).astype('f')
    logger.info("========== Start training ==========")
    logger.info("Iteration Loss")
    for schedule in param_update_schedule:
        htfd.fit(target_spectrogram,
                 n_iter=schedule["n_iter"],
                 update_flags=schedule["update_flags"],
                 post_process=postproc,
                 log_interval=log_interval)
    logger.info("========== Finish training =========")
    del target_spectrogram
    if gpu >= 0:
        htfd.to_cpu()
    return htfd


if __name__ == "__main__":
    numpy.seterr(invalid="raise")
    import utils.evaluate as evaluate_module
    main()
    temp_dir = evaluate_module.octave.temp_dir
    evaluate_module.octave.exit()
    import shutil
    logger.info(f"remove {temp_dir}")
    shutil.rmtree(temp_dir)
