#!/usr/bin/env python

import pathlib

try:
    from setuptools import setup, find_packages
except:
    raise Exception("setuptools is required for installation")

VERSION_PATH = pathlib.Path(__file__).parent / "target_finder" / "version.py"


def get_version():
    """Get the version number without running version.py.
    Returns:
        str: The current uavaustin-target-finder version.
    """

    with VERSION_PATH.open("r") as version:
        out = {}
        exec(version.read(), out)
        return out["__version__"]


setup(
    name="target-finder",
    version=get_version(),
    author="UAV Austin",
    url="https://github.com/uavaustin/target-finder",
    packages=find_packages(),
    install_requires=["Pillow>=4.3.0", "scipy==1.4.1", "scikit-learn==0.22.2.post1"],
    entry_points={"console_scripts": ["target-finder-cli=target_finder.cli:run"]},
    license="MIT",
)
