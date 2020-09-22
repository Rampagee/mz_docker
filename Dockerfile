
# ===========================================================
# Copyright(C) 2015-2019 Mipsology SAS.  All Rights Reserved.
# ===========================================================

FROM ubuntu:bionic as zebra_base

ENV DEBIAN_FRONTEND=noninteractive

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
  rsync \
  cython3 \
  python3-pip \
  swig \
  python3-tk

RUN apt-get update && apt-get install -y --no-install-recommends caffe-cpu

RUN python3 -m pip --no-cache-dir install --upgrade pip

# install basic dependencies globally

RUN python3 -m pip --no-cache-dir install opencv-python==4.1.1.26 pillow==5.1.0

RUN python3 -m pip --no-cache-dir install tensorflow==1.14.0 numpy==1.15.0
# dependencies for pytorch
RUN python3 -m pip --no-cache-dir install future==0.18.1        \
                                          h5py==2.7.1           \
                                          cffi==1.13.1          \
                                          pybind11==2.4.3       \
                                          onnx==1.5.0

# install torch without gpu support to reduce docker image size
RUN python3 -m pip --no-cache-dir install torch==1.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# install run_classification dependencies
RUN python3 -m pip --no-cache-dir install lxml==4.4.2

RUN python3 -m pip --no-cache-dir install tqdm==4.43.0          \
                                          slidingwindow==0.0.13 \
                                          pycocotools==2.0.0


RUN echo "root:demo" | chpasswd

######################################################################
#
# all RUN command should start with umask 0, like
# RUN umask 0 &&
#
# NOTE: if the current build is changed, it will invalidate the cache
# for all below builds
#
######################################################################

ENV PYTHONPATH .local/lib/python3.6/site-packages:

WORKDIR /home/demo

######################################################################
# if an application requires a specific python package, it has to install it
# locally (using PYTHONUSERBASE=.local), then the package will be used through
# PYTHONPATH

######################################################################
# all application should follow the following:
# FROM zebra_base AS <app_name>
# - git clone
# - PYTHONUSERBASE=.local python3 -m pip install --user <dependencies>
# - download weights
# - freeze models (if applicable)
# - copy patch and apply
#
# NOTE: for optimal docker cache, the copy patch should come at the end
#
######################################################################



######################################################################
# DEMO:darkflow yoloV2 from https://github.com/thtrieu/darkflow
######################################################################
FROM zebra_base AS darkflow

RUN umask 0 && \
  git clone https://github.com/thtrieu/darkflow && \
  cd darkflow && \
  git checkout -b mipso b2aee0000cd2a956b9f1de6dbfef94d53158b7d8

RUN umask 0 && \
  cd darkflow && \
  mkdir -p download && \
  cd download && \
  echo "Download weights" && \
  wget https://github.com/pjreddie/darknet/raw/master/cfg/yolov2-voc.cfg && \
  wget https://pjreddie.com/media/files/yolov2-voc.weights && \
  wget https://github.com/pjreddie/darknet/raw/master/cfg/yolov2.cfg && \
  wget https://pjreddie.com/media/files/yolov2.weights && \
  tail -c +5 yolov2-voc.weights > yolov2-voc_fixed.weights && \
  rm yolov2-voc.weights

# to merge with above layer when base depencency will be broken
RUN umask 0 && \
  cd darkflow && \
  cd download && \
  echo "Download weights" && \
  wget https://github.com/pjreddie/darknet/raw/master/cfg/yolov2-tiny.cfg && \
  wget https://pjreddie.com/media/files/yolov2-tiny.weights && \
  tail -c +5 yolov2-tiny.weights > yolov2-tiny_fixed.weights && \
  rm yolov2-tiny.weights

COPY darkflow.patch /tmp/darkflow.patch

RUN umask 0 && \
  cd darkflow && \
  patch -p1 -i /tmp/darkflow.patch

RUN umask 0 && \
  cd darkflow && \
  python3 setup.py build_ext --inplace && \
  python3 ./flow --model download/yolov2-voc.cfg --load download/yolov2-voc_fixed.weights --labels /dev/null --savepb ; test -f built_graph/yolov2-voc.pb && \
  python3 ./flow --model download/yolov2.cfg --load download/yolov2.weights --labels cfg/coco.names --savepb ; test -f built_graph/yolov2.pb && \
  python3 ./flow --model download/yolov2-tiny.cfg --load download/yolov2-tiny_fixed.weights --labels cfg/coco.names --savepb ; test -f built_graph/yolov2-tiny.pb && \
  rm -rf download


######################################################################
# DEMO:tensorflow-yolo3 yoloV3 from https://github.com/aloyschen/tensorflow-yolo3
######################################################################
FROM zebra_base AS tensorflow_yolov3

RUN umask 0 && \
  git clone https://github.com/aloyschen/tensorflow-yolo3 && \
  cd tensorflow-yolo3 && \
  git checkout -b mipso 646f4532487ff728695c55fb3b9a29fcd631e68d

RUN umask 0 && \
  cd tensorflow-yolo3 && \
  echo "Download weights" && \
  cd model_data && \
  wget https://pjreddie.com/media/files/yolov3.weights

COPY tensorflow-yolo3.patch /tmp/tensorflow-yolo3.patch

RUN umask 0 && \
  cd tensorflow-yolo3 && \
  patch -p1 -i /tmp/tensorflow-yolo3.patch

RUN umask 0 && \
  cd tensorflow-yolo3 && \
  python3 detect.py --save_pb --out_file /dev/null --image_file dog.jpg && \
  rm model_data/yolov3.weights


######################################################################
# DEMO:pytorch_yolo2 pytorch yoloV2 from https://github.com/marvis/pytorch-yolo2
######################################################################
FROM zebra_base AS pytorch_yolo2

RUN umask 0 && \
  git clone https://github.com/marvis/pytorch-yolo2 && \
  cd pytorch-yolo2 && \
  git checkout -b mipso 6c7c1561b42804f4d50d34e0df220c913711f064

RUN umask 0 && \
  cd pytorch-yolo2 && \
  wget http://pjreddie.com/media/files/yolo.weights

COPY pytorch-yolo2.patch /tmp/pytorch-yolo2.patch

RUN umask 0 && \
  cd pytorch-yolo2 && \
  patch -p1 -i /tmp/pytorch-yolo2.patch


######################################################################
# DEMO:yolo2_pytorch pytorch yoloV2 from https://github.com/longcw/yolo2-pytorch
######################################################################
FROM zebra_base AS yolo2_pytorch

RUN umask 0 && \
  git clone https://github.com/longcw/yolo2-pytorch && \
  cd yolo2-pytorch && \
  git checkout -b mipso 17056ca69f097a07884135d9031c53d4ef217a6a

COPY wget_google_drive.bash /tmp/wget_google_drive.bash

RUN umask 0 && \
  cd yolo2-pytorch && \
  mkdir -p models && \
  /tmp/wget_google_drive.bash https://drive.google.com/uc?id=0B4pXCfnYmG1WUUdtRHNnLWdaMEU models/yolo-voc.weights.h5 && \
  [ "$( stat -c '%s' models/yolo-voc.weights.h5 )" = "268676308" ]


COPY yolo2-pytorch.patch /tmp/yolo2-pytorch.patch

RUN umask 0 && \
  cd yolo2-pytorch && \
  patch -p1 -i /tmp/yolo2-pytorch.patch

RUN umask 0 && \
  cd yolo2-pytorch && \
  ./make.sh

######################################################################
# DEMO:VDSR-Tensorflow VDSR from https://github.com/DevKiHyun/VDSR-Tensorflow
######################################################################
FROM zebra_base AS vdsr_tensorflow

RUN umask 0 && \
  git clone https://github.com/DevKiHyun/VDSR-Tensorflow && \
  cd VDSR-Tensorflow && \
  git checkout -b mipso 117bbb2c4b35daadbba9b29a7c8a4e674b0cd349

COPY VDSR-Tensorflow.patch /tmp/VDSR-Tensorflow.patch

RUN umask 0 && \
  cd VDSR-Tensorflow && \
  patch -p1 -i /tmp/VDSR-Tensorflow.patch

RUN umask 0 && \
  cd VDSR-Tensorflow/VDSR && \
  echo "Freeze model" && \
  python3 demo.py --save_pb

RUN umask 0 && \
  cd VDSR-Tensorflow/VDSR && \
  echo "[runOptimization]\n\tnetworkOptimizers=NeutralMaxpoolAsInput:BOTH,NeutralMaxpoolOnAllElementWise:BOTH,EltWiseBalance:BOTH,ConcatBalance:RUN" > zebra.ini && \
  echo "[debug]\n\teltWiseType=ELTWISE_OPTIMIZED_OUT" >> zebra.ini && \
  echo "[quantization]\n\tminimalBatchSize=1" >> zebra.ini

######################################################################
# DEMO:Tensorflow-YOLOv3 from https://github.com/kcosta42/Tensorflow-YOLOv3
######################################################################
FROM zebra_base AS tensorflow_yolov3_tiny

RUN umask 0 && \
  git clone https://github.com/kcosta42/Tensorflow-YOLOv3 && \
  cd Tensorflow-YOLOv3  && \
  git checkout -b mipso e986a8c5a538935106c7435b05bf4336b1bdb721

RUN umask 0 && \
  cd Tensorflow-YOLOv3  && \
  cd weights && \
  wget https://pjreddie.com/media/files/yolov3.weights && \
  wget https://pjreddie.com/media/files/yolov3-tiny.weights && \
  cd .. && \
  python3 convert_weights.py && \
  python3 convert_weights.py --tiny && \
  rm -f weights/yolov3.weights weights/yolov3-tiny.weights

COPY Tensorflow-YOLOv3.patch /tmp/Tensorflow-YOLOv3.patch

RUN umask 0 && \
  cd Tensorflow-YOLOv3 && \
  patch -p1 -i /tmp/Tensorflow-YOLOv3.patch

RUN umask 0 && \
  cd Tensorflow-YOLOv3 && \
  echo "Freezing YOLOv3 model" && \
  python3 detect.py image 0.5 0.5 data/images/dog.jpg --freeze && \
  echo "Freezing YOLOv3_tiny model" && \
  python3 detect.py image 0.5 0.5 --tiny data/images/dog.jpg --freeze && \
  rm -rf weights

RUN umask 0 && \
  cd Tensorflow-YOLOv3 && \
  echo "[quantization]\n\tminimalBatchSize=1" > zebra.ini

######################################################################
# DEMO:tf-pose-estimation from https://github.com/ildoonet/tf-pose-estimation
######################################################################
FROM zebra_base AS tf-pose-estimation

RUN umask 0 && \
  git clone https://github.com/ildoonet/tf-pose-estimation && \
  cd tf-pose-estimation && \
  git checkout -b mipso 9aaf34708e9863eeab9685a7ec52a45f3b576e2d

RUN umask 0 && \
  cd tf-pose-estimation && \
  cd tf_pose/pafprocess && \
  swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace

RUN umask 0 && \
  cd tf-pose-estimation && \
  sh ./models/graph/cmu/download.sh

COPY tf-pose-estimation.patch /tmp/tf-pose-estimation.patch

RUN umask 0 && \
  cd tf-pose-estimation && \
  patch -p1 -i /tmp/tf-pose-estimation.patch

RUN umask 0 && \
  cd tf-pose-estimation && \
  echo "[quantization]\n\tminimalBatchSize=1" >> zebra.ini

######################################################################
# DEMO:ALL all demo
######################################################################
FROM zebra_base AS all_common

# now copy all examples

COPY --from=darkflow            /home/demo/ /home/demo/
COPY --from=tensorflow_yolov3   /home/demo/ /home/demo/
COPY --from=pytorch_yolo2       /home/demo/ /home/demo/
COPY --from=yolo2_pytorch       /home/demo/ /home/demo/
COPY --from=vdsr_tensorflow     /home/demo/ /home/demo/
COPY --from=tensorflow_yolov3_tiny /home/demo/ /home/demo/
COPY --from=tf-pose-estimation  /home/demo/ /home/demo/


FROM all_common AS all

# create local user for outside access (X and volumes)

ARG UID=1000
RUN adduser demo --gecos "" --disabled-password --no-create-home --uid $UID && \
  chown demo:demo /home/demo && \
  echo "demo:demo" | chpasswd

USER demo


COPY README /tmp/README
RUN grep -v  '^#' /tmp/README | grep -v '^$' > .bash_history

# Zebra release will be mounted in /home/demo/zebra
RUN mkdir -p ~/.mipsology/zebra && echo "[runSession]\n\tdirectory=/home/demo/zebra/examples/models.zebra" > ~/.mipsology/zebra/zebra.ini

