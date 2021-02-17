# HTFD: Harmonic-temporal factor decomposition for unsupservised monaural source separation of harmonic sounds

[**Paper**](https://doi.org/10.1109/TASLP.2020.3037487)
| [**Demo**](https://tomohikonakamura.github.io/Tomohiko-Nakamura/demo/HTFD/)

*Harmonic-temporal factor decomposition (HTFD)* is an unsupervised source separation method of harmonic sounds.
This method can be also applied to music editing systems (e.g., changing from major to minor scales).

HTFD encompasses the well-known three concepts cultivated in the audio signal processing community (computational auditory scene analysis, non-negative matrix factorization, and source-filter model).
Please see our paper [1] for the details.

# How to run
**Caution: We check only the GPU implementation. The CPU implementaion is now under construction and not optimized. The CPU implementation may return different results from the GPU version.**

## Requirements
We checked that the codes work with the Linux environment:
- TITAN RTX (24GB RAM)
- docker-ce (Docker version 20.10.3, build 48d30b5)
  - Host's nvidia driver version: 430.6
- Octave
- Python 3.8
  ```
  cupy==8.4.0
  coloredlogs==15.0
  librosa==0.8.0
  matplotlib==3.3.4
  tqdm==4.56.0
  pyyaml==5.4.1
  oct2py==5.2.0
  ```

We recommend the following Docker-based installation for reproducibility.

### Docker
- Install Docker. (If installed, skip this step.)
- Clone this repository and `cd` to it.
- Download the evaluation tool.
  ```
  wget http://bass-db.gforge.inria.fr/bss_eval/bss_eval_2.1.zip && unzip bss_eval_2.1.zip
  ```
- Build and run a container.
  ```bash
  docker build -t htfd .
  docker run --rm -it --gpus all -v "${PWD}:/opt/htfd" htfd:latest bash
  ```
  If `docker-compose` is installed,
  ```bash
  docker-compose build
  docker-compose run --rm htfd
  ```
- Check whether the codes work.
  ```bash
  cd /opt/htfd
  python key_transpose.py -h
  ```

# Examples
## Separate synthetic data
- Download the synthetic dataset and agree the conditions of use from [here](https://docs.google.com/forms/d/e/1FAIpQLSc4lO5FieU6GBT0_z-b9weorHaKn5M0GrcChAmp3P8ibemlYQ/viewform?usp=sf_link). (You can use this dataset **only for academic research projects** and cannot use for any commercial purpose.)
  - This dataset consists of MIDI-synthesized audio signals of 7 music excerpts. These excerpts were obtained from Classic Music No. 1 to 7 from the RWC music database [3].
  
- Extract the downloaded file to `/path/to/repository/synth_data`.
- Execute separation and evaluation codes for HTFD and SF-HTFD.
  ```bash
  cd /opt/htfd
  # HTFD
  python unsupervised_exp.py -i synth_data/RM-C001 -o results --gpu 0 --methods HTFD --alpha_Us -5 --log_interval 1 --tf_conf conf.yaml
  # SF-HTFD
  python unsupervised_exp.py -i synth_data/RM-C001 -o results --gpu 0 --methods SFHTFD --alpha_Us -2 --log_interval 1 --tf_conf conf.yaml --n_filters 3 --filter_deg 48
  ```
- You can get the csv files of the evaluation result at `results/{HTFD,SFHTFD}_log10alphaU*/RM-C001/summary.txt`.

## Musical key transposition
- Download the Bach10 dataset (please see http://www2.ece.rochester.edu/projects/air/resource.html).
- Unzip the downloaded zip file and extract it as `/path/to/repository/real_data`.
- Execute key transposition by HTFD and SF-HTFD.
  ```bash
  cd /opt/htfd
  # HTFD
  python key_transpose.py -i real_data/01-AchGottundHerr/01-AchGottundHerr.wav --keys "C Major" -o results/real --tf_conf conf.yaml --methods HTFD4Real --gpu 0 --normalize_output
  # SFHTFD
  python key_transpose.py -i real_data/01-AchGottundHerr/01-AchGottundHerr.wav --keys "C Major" -o results/real --tf_conf conf.yaml --methods SFHTFD --gpu 0 --normalize_output
  ```

# Cite
```
@article{TNakamura2020IEEEACMTASLP,
 author={Nakamura, Tomohiko and Kameoka, Hirokazu},
 journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
 title = {Harmonic-Temporal Factor Decomposition for Unsupervised Monaural Separation of Harmonic Sounds},
 year=2020,
 month=nov,
 volume={29},  
 number={},  
 pages={68--82},  
 doi={10.1109/TASLP.2020.3037487}
}
```
If you use only the `pyfacwt` module, please cite [2]. This module performs the fast approximate version of the continuous wavelet transform and its inverse transform.

# License
[MIT License](LICENSE)

# References
[1] Tomohiko Nakamura and Hirokazu Kameoka, "Harmonic-Temporal Factor Decomposition for Unsupervised Monaural Separation of Harmonic Sounds," IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 68–82, Nov. 2020.  
[2] Tomohiko Nakamura and Hirokazu Kameoka, "Fast Signal Reconstruction from Magnitude Spectrogram of Continuous Wavelet Transform Based on Spectrogram Consistency," in Proceedings of the 17th International Conference on Digital Audio Effects, Sep. 2014, pp. 129–135.   
[3] Masataka Goto, "Development of the RWC Music Database," Proceedings of the 18th International Congress on Acoustics, pp.I-553-556, Apr. 2004.

