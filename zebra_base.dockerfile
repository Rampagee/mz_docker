
# ===========================================================
# Copyright(C) 2015-2019 Mipsology SAS.  All Rights Reserved.
# ===========================================================

FROM ubuntu:bionic

# install package dependencies
RUN apt-get update && apt-get install -y \
  git \
  vim \
  wget \
  xterm \
  x11-xserver-utils \
  libglib2.0-dev \
  libsm-dev \
  libxrender-dev \
  cython \
  python-pip \
  cython3 \
  python3-pip


RUN pip3 install --upgrade pip

RUN pip3 --no-cache-dir install opencv-python
RUN pip3 --no-cache-dir install tensorflow==1.8.0
#RUN pip3 --no-cache-dir install torch pillow cffi h5py onnx pybind11 future


RUN echo "root:demo" | chpasswd

# create local user for outside access (X and volumes)

ARG UID=1000
RUN adduser demo --gecos "" --disabled-password --uid $UID && \
  echo "demo:demo" | chpasswd

USER demo

# Zebra release will be mounted in /home/demo/zebra
RUN mkdir -p ~/.mipsology/zebra && echo "[runSession]\n\tdirectory=/home/demo/zebra/examples/models" > ~/.mipsology/zebra/zebra.ini

