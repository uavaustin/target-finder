# target-finder

[![Build Status](
    http://ci.uavaustin.org/job/target-finder-smoke/badge/icon)](
    http://ci.uavaustin.org/view/Image%20Recognition/)

Python library for identifying AUVSI SUAS targets

## Installation

To install from source, try this:

```sh
$ pip install git+https://github.com/uavaustin/target-finder.git
```

This should be able to get the OpenCV dependency as well. If you already have
an OpenCV installation (that was not installed by `pip install opencv-python`)
this might cause problems.

Installs from source will not contain trained models, you'll need to train
before recognizing targets.

## Using the GPU

By default the library requires the CPU-only version of Tensorflow. If you'd
like to run it with the GPU, uninstall the version this one installed and
replace it with the GPU version like so

```sh
$ pip uninstall tensorflow
$ pip install tensorflow-gpu
```

*Check [here](https://www.tensorflow.org/install/) to make sure you have the
requirements for using the GPU.*

# Testing
The target-finder library uses [tox](https://github.com/tox-dev/tox) to manage
its tests. Both smoke and end-to-end tests are handled by tox separately.

To run the tests, first install tox.

```sh
$ pip install tox
```

Now smoke and end-to-end tests can be run by simply calling

```sh
$ tox
```

in the main directory.

To run only the smoke or end-to-end tests use the following:

```sh
$ tox -e smoke
$ tox -e e2e
```

To run end-to-end tests without training, use the following:

```sh
$ tox -e e2e-no-train
```
