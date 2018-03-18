# target-finder

Python library for identifying AUVSI SUAS targets.

## Installation

To install from source right from GitHub, run the following:

```sh
$ pip install git+https://github.com/uavaustin/target-finder.git
```

Python 3 is required. *If `python --version` shows Python 2, then use `python3`
and `pip3` instead.*

This will not install OpenCV and Tensorflow automatically, those must be
installed separately. Other dependencies, however, will be fetched
automatically.

The easiest way to install OpenCV is with `pip install opencv-python`. However,
this [might not work on all platforms](
    https://github.com/skvark/opencv-python/issues/13).

To get Tensorflow, you can either run `pip install tensorflow` or
`pip install tensorflow-gpu`. The GPU version can only be used if the
[Tensorflow GPU requirements](
    https://www.tensorflow.org/install/install_linux#NVIDIARequirements) are
met.

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
