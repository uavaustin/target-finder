FROM uavaustin/target-finder-model-env:tf1

WORKDIR /

RUN apt-get update \
    && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*


COPY . target-finder

WORKDIR target-finder

RUN git clone https://github.com/tensorflow/tensorrt.git

WORKDIR tensorrt/tftrt/examples/object_detection

RUN git submodule update --init \
  && ./install_dependencies.sh 

WORKDIR /target-finder/tensorrt

RUN python setup.py install --user
