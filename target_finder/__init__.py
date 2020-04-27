"""Entrypoint for the target_finder library."""

from target_finder.classification import find_targets, load_models
from target_finder.types import Color, Shape, Target
from target_finder.version import __version__
