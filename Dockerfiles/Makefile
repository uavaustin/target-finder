# Flags for docker when building images, meant to be overridden
DOCKERFLAGS :=

IMAGE_NAME := uavaustin/target-finder-env

IMAGE_TAG := latest

MOUNT_DIR := /

CUR_DIR := $(shell pwd)

.PHONY: all clean
all: image

.PHONY: image
image:
	docker build -t $(IMAGE_NAME):latest -f Dockerfile $(DOCKERFLAGS) ..

.PHONY: xavier
xavier:
	docker build -t $(IMAGE_NAME):xavier -f Dockerfile.xavier $(DOCKERFLAGS) ..
