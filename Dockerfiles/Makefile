# Flags for docker when building images, meant to be overridden
DOCKERFLAGS :=

IMAGE_NAME := uavaustin/target-finder-model-env

IMAGE_TAG := latest

MOUNT_DIR := /

CUR_DIR := $(shell pwd)

.PHONY: all clean
all: image

.PHONY: image
image:
	docker build --no-cache -t $(IMAGE_NAME):tf -f Dockerfile $(DOCKERFLAGS) ..
