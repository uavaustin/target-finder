"""Entrypoint for the target_finder library."""

from .classification import find_targets, load_models
from .types import Color, Shape, Target
from .version import __version__
