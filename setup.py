#!/usr/bin/env python

import os

try:
    from setuptools import setup, find_packages
except:
    raise Exception('setuptools is required for installation')


def join(*paths):
    """Join and normalize several paths.
    Args:
        *paths (List[str]): The paths to join and normalize.
    Returns:
        str: The normalized path.
    """

    return os.path.normpath(os.path.join(*paths))


VERSION_PATH = join(__file__, '..', 'target_finder', 'version.py')
REQUIREMENTS_PATH = join(__file__, '..', 'requirements.txt')


def get_version():
    """Get the version number without running version.py.
    Returns:
        str: The current uavaustin-target-finder version.
    """

    with open(VERSION_PATH, 'r') as version:
        out = {}

        exec(version.read(), out)

        return out['__version__']


def load_requirements():
    """Get the requirements file in this directory.
    Returns:
        List[str]: A list of pip readable strings.
    """

    req = []
    actual = []

    with open(REQUIREMENTS_PATH, 'r') as requirements:
        req = requirements.read().split('\n')

    for r in req:
        if 'tensorflow' in r:
            # If the environment variable to use the GPU-enabled
            # Tensorflow library is set, install it instead
            if os.environ.get('USE_GPU_TENSORFLOW', '0') != '0':
                actual.append(r.replace('tensorflow', 'tensorflow-gpu'))
            else:
                actual.append(r)

        elif 'opencv-python' in r:
            # If we can already import OpenCV we don't shouldn't try
            # to install it
            try:
                import cv2
            except ImportError as e:
                actual.append(r)

        else:
            actual.append(r)

    return actual


setup(
    name='uavaustin-target-finder',
    version=get_version(),
    author='Unmanned Aerial Vehicle Team | UT Austin',
    url='https://github.com/uavaustin/target-finder',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[load_requirements()],
    entry_points='''
        [console_scripts]
        train=target_finder.cli:do_train
        get_targets=target_finder.cli:do_get_targets
    ''',
    license='MIT'
)
