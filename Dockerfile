FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

WORKDIR /

COPY requirements.txt requirements.txt

RUN apt update && \
    apt install -y git

ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda

RUN python -m pip install --upgrade pip 
# RUN pip install torchlars
RUN pip install -r requirements.txt --use-feature=2020-resolver
RUN pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git 

COPY . .