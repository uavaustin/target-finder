"""Entrypoint for the target_finder library."""

from .classification import find_targets
from .preprocessing import find_blobs
from .training import train
from .types import Blob, Color, Shape, Target
from .version import __version__
