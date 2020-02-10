FROM nvcr.io/nvidia/tensorflow:19.10-py3

WORKDIR /

RUN apt-get update \
    && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-gpu.txt .
RUN pip3 install -r requirements-gpu.txt \
https://github.com/uavaustin/target-finder-model/releases/download/v1.0.0/target-finder-model-1.0.0.tar.gz

WORKDIR /host