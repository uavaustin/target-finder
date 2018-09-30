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


def get_version():
    """Get the version number without running version.py.
    Returns:
        str: The current uavaustin-target-finder version.
    """

    with open(VERSION_PATH, 'r') as version:
        out = {}

        exec(version.read(), out)

        return out['__version__']


setup(
    name='target-finder',
    version=get_version(),
    author='UAV Austin',
    url='https://github.com/uavaustin/target-finder',
    packages=find_packages(),
    install_requires=[
        'Pillow>=4.3.0',
        'scipy',
        'webcolors>=1.7',
        'scikit-learn'
    ],
    entry_points='''
        [console_scripts]
        target-finder-cli=target_finder.cli:run
    ''',
    license='MIT'
)
