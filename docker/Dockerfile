FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y \
    && apt-get install -y \
    make \
    libgl1-mesa-dev \
    libglib2.0-0 \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*
    

COPY requirements.txt /home/multi-label-weather-classification/requirements.txt

WORKDIR /home/multi-label-weather-classification

RUN pip install -r requirements.txt
