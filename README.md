# target-finder

Python library for identifying AUVSI SUAS targets.

## Installation

To install from source right from GitHub, run the following:

```sh
$ pip install git+https://github.com/uavaustin/target-finder.git
```

Alternatively, target-finder can be fetched from GitHub Releases:

```sh
$ pip install https://github.com/uavaustin/target-finder/releases/download/v0.1.1/target-finder-0.1.1.tar.gz
```

Python 3 is required. *If `python --version` shows Python 2, then use `python3`
and `pip3` instead.*

This will not install OpenCV, Tensorflow, and target-finder-model
automatically, those must be installed separately. Other dependencies, however,
will be fetched automatically.

### OpenCV

The easiest way to install OpenCV is with `pip install opencv-python`. However,
this [might not work on all platforms](
    https://github.com/skvark/opencv-python/issues/13).

### Tensorflow

To get Tensorflow, you can either run `pip install tensorflow` or
`pip install tensorflow-gpu`. The GPU version can only be used if the
[Tensorflow GPU requirements](
    https://www.tensorflow.org/install/install_linux#NVIDIARequirements) are
met.

### target-finder-model

The model used for target-finder is packaged in target-finder-model. It can be
installed from GitHub Releases:

```sh
$ pip install https://github.com/uavaustin/target-finder-model/releases/download/v0.1.0/target-finder-model-0.1.0.tar.gz
```

## Command-line Interface

The library ships with `target-finder-cli` command, type `target-finder-cli -h`
in the command-line after installing for help and usage.

## Testing

To run the tests you'll first need to install install pytest if it is not
already installed: `pip install pytest`.

The OpenCV and Tensorflow depenedencies must also be installed per the
installation instructions.

Before running the tests, you'll need to install the target-finder library with
`pip`. The best way to do it is with the `-e` flag so you can make changes to
the library without needing to reinstall each time you edit a file. Afterwards,
just use `pytest` to run the tests.

```
# Should only need to do this once.
$ pip install -e .

# Run the test suite.
$ pytest
```
