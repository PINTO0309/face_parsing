FROM pinto0309/base-x64-ubuntu2204-cuda118:latest
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y \
        python3-all-dev \
        python-is-python3 \
        python3-pip \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgtk2.0-dev \
        libgl1-mesa-dev \
        git \
        git-lfs \
        curl \
        wget \
        sudo \
        pkg-config

RUN pip install -U pip \
    && pip install opencv-python==4.8.0.74 \
    && pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu118 \
    && pip install onnx==1.13.1 \
    && pip install onnxruntime-gpu==1.15.1 \
    && pip install onnxsim==0.4.33 \
    && pip install nvidia-pyindex \
    && pip install onnx_graphsurgeon \
    && pip install onnx2tf \
    && pip install simple_onnx_processing_tools \
    && pip install tensorflow==2.12.0 \
    && pip install protobuf==3.20.3 \
    && pip install h5py==3.7.0 \
    && pip install psutil==5.9.5

ENV USERNAME=user
RUN echo "root:root" | chpasswd \
    && adduser --disabled-password --gecos "" "${USERNAME}" \
    && echo "${USERNAME}:${USERNAME}" | chpasswd \
    && echo "%${USERNAME}    ALL=(ALL)   NOPASSWD:    ALL" >> /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}
USER ${USERNAME}
ARG WKDIR=/workdir
WORKDIR ${WKDIR}
RUN sudo chown ${USERNAME}:${USERNAME} ${WKDIR}