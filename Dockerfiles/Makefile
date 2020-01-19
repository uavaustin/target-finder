# Flags for docker when building images, meant to be overridden
DOCKERFLAGS :=

IMAGE_NAME := uavaustin/target-finder-model-env

IMAGE_TAG := latest

MOUNT_DIR := /

CUR_DIR := $(shell pwd)

.PHONY: all clean
all: image, test

.PHONY: image
image:
	docker build -t $(IMAGE_NAME):tf -f Dockerfile $(DOCKERFLAGS) ..

.PHONY: test
test:
	docker build --no-cache -t $(IMAGE_NAME):test -f Dockerfile.test $(DOCKERFLAGS) ..
	docker run --rm -v $(CUR_DIR)/../:/host $(IMAGE_NAME):test $(DOCKERARGS)  

.PHONY: xavier
xavier:
	docker build -t $(IMAGE_NAME):xavier -f Dockerfile.xavier $(DOCKERFLAGS) ..
