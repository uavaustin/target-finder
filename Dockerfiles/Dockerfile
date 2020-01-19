FROM uavaustin/target-finder-model-env:tf1

WORKDIR /

RUN apt-get update \
    && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

COPY . target-finder

RUN pip3 install https://github.com/uavaustin/target-finder-model/releases/download/v1.0.0/target-finder-model-v1.0.0.tar.gz
