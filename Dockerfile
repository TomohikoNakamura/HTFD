FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3.8 python3.8-dev python3.8-distutils \
    curl libfftw3-dev libboost-python-dev libboost-numpy-dev \
    ffmpeg libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Install pip
RUN rm /usr/bin/python && ln -s /usr/bin/python3.8 /usr/bin/python && curl -O https://bootstrap.pypa.io/get-pip.py && python get-pip.py && rm get-pip.py

RUN pip install --no-cache-dir cupy-cuda101==8.4.0 coloredlogs==15.0 librosa==0.8.0 matplotlib==3.3.4 tqdm==4.56.0 pyyaml==5.4.1

# For octave
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends octave \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* && pip install --no-cache-dir oct2py==5.2.0

# Install pyfacwt
COPY facwt /opt/facwt
RUN cd /opt/facwt && pip install -e . --no-cache-dir

