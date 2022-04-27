FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04

ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install python3.8
RUN : \
    && apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:deadsnakes \
    && apt-get install -y --no-install-recommends python3.8-venv \
    && apt-get install libpython3.8-de -y \
    && apt-get install python3.8-dev -y \
    && apt-get install build-essential -y \
    && apt-get clean \
    && :
    
# Add env to PATH
RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH

# Install ASAP
RUN : \
    && apt-get update \
    && apt-get -y install curl git \
    && curl --remote-name --location "https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.0-(Nightly)/ASAP-2.0-py38-Ubuntu2004.deb" \
    && dpkg --install ASAP-2.0-py38-Ubuntu2004.deb || true \
    && apt-get -f install --fix-missing --fix-broken --assume-yes \
    && ldconfig -v \
    && apt-get clean \
    && echo "/opt/ASAP/bin" > /venv/lib/python3.8/site-packages/asap.pth \
    && rm ASAP-2.0-py38-Ubuntu2004.deb \
    && :


# Libraries
RUN python -m pip install -U pip
RUN pip install git+https://github.com/DIAGNijmegen/pathology-whole-slide-data@216acd8a9166246f33c5996080e96a4d03e44402 
RUN pip install tensorflow-gpu==2.5
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install numpy==1.20.0 
RUN pip install albumentations 
RUN pip install pycm 
RUN pip install tqdm
RUN pip install matplotlib
RUN pip install segmentation-models-pytorch

RUN apt install nvidia-modprobe -y

# FOLDERS and PERMISSIONS
RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output /tempoutput /tempoutput/segoutput /tempoutput/detoutput /tempoutput/bulkoutput \
    /configs \
    && chown algorithm:algorithm /opt/algorithm /input /output /tempoutput /tempoutput/segoutput /tempoutput/detoutput \
    /tempoutput/bulkoutput

USER algorithm
WORKDIR /opt/algorithm
ENV PATH="/home/algorithm/.local/bin:${PATH}"

# add scripts and models 
ADD --chown=algorithm:algorithm models /opt/algorithm/
ADD --chown=algorithm:algorithm configs /opt/algorithm/
ADD --chown=algorithm:algorithm weights /opt/algorithm/weights
ADD --chown=algorithm:algorithm efficientnet /opt/algorithm/efficientnet
COPY --chown=algorithm:algorithm write_annotations.py /opt/algorithm/
COPY --chown=algorithm:algorithm write_mask.py /opt/algorithm/
COPY --chown=algorithm:algorithm detection_inference.py /opt/algorithm/
COPY --chown=algorithm:algorithm segmentation_inference.py /opt/algorithm/
COPY --chown=algorithm:algorithm concave_hull.py /opt/algorithm/
COPY --chown=algorithm:algorithm utils.py /opt/algorithm/
COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm association.py /opt/algorithm/
COPY --chown=algorithm:algorithm nms.py /opt/algorithm/
COPY --chown=algorithm:algorithm rw.py /opt/algorithm/
COPY ./testinput/ /input/
# RUN chown -R user /input/

ENTRYPOINT python -u -m process $0 $@s

## ALGORITHM LABELS ##
# These labels are required
LABEL nl.diagnijmegen.rse.algorithm.name=bcsegdetrumc

# These labels are required and describe what kind of hardware your algorithm requires to run.
LABEL processor.cpus="4"
LABEL processor.cpu.capabilities="null"
LABEL processor.memory="30G"
LABEL processor.gpu_count="1"
LABEL processor.gpu.compute_capability="null"
LABEL processor.gpu.memory="15G"