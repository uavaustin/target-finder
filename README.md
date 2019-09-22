# target-finder

> Python library for identifying AUVSI SUAS targets.

[![Build Status](https://travis-ci.org/uavaustin/target-finder.svg?branch=master)](https://travis-ci.org/uavaustin/target-finder)
[![Test Coverage](https://coveralls.io/repos/github/uavaustin/target-finder/badge.svg?branch=master)](https://coveralls.io/github/uavaustin/target-finder?branch=master)

## Installation

To install from source right from GitHub, run the following:

```sh
$ pip install git+https://github.com/uavaustin/target-finder.git
```

Alternatively, target-finder can be fetched from GitHub Releases:

```sh
$ pip install https://github.com/uavaustin/target-finder/releases/download/v0.3.1/target-finder-0.3.1.tar.gz
```

Python 3 is required. *If `python --version` shows Python 2, then use `python3`
and `pip3` instead.*

This will not install OpenCV and target-finder-model automatically,
those must be installed separately. Other dependencies, however,
will be fetched automatically.

### OpenCV

The easiest way to install OpenCV is with `pip install opencv-python`. However,
this [might not work on all platforms](
    https://github.com/skvark/opencv-python/issues/13).

### target-finder-model

The model used for target-finder is packaged in target-finder-model. It can be
installed from GitHub Releases:

```sh
$ pip install https://github.com/uavaustin/target-finder-model/releases/download/v0.2.0/target-finder-model-0.2.0.tar.gz
```

## Command-line Interface

The library ships with `target-finder-cli` command, type `target-finder-cli -h`
in the command-line after installing for help and usage.

```text
usage: target-finder-cli [-h] [-v] targets ...

optional arguments:
  -h, --help       show this help message and exit
  -v, --version    show the version and exit

subcommands:
    targets        finds the targets in images
```

For example, to check for all the targets in two folders and put them in a
folder called `out/`, you can use

```sh
$ target-finder-cli targets folder-1 folder-2 -o out
```

By default, all the target images and metadata will go into your current
directory.

## Testing

The target-finder library uses [tox](https://github.com/tox-dev/tox) to manage
its tests.

To run the tests, first install tox.

```sh
$ pip install tox
```

Now unit tests can be run by simply calling:

```sh
$ tox
```

This will take care of installing the "standard" opencv-python, target-finder,
and target-finder-model packages. Note that this may not work out-of-the-box
on all systems. The tests can be run manually by fetching the test dependencies
needed (see `tox.ini`) and run with `pytest`.
