"""Contains the Color enum for identifying colors in targets."""

from enum import Enum


class Color(Enum):
    """Contains colors for the AUVSI SUAS Interop Server.

    These colors can be used for both the background color and the
    alphanumeric color.

    Note that Color.NONE can be used if a color cannot be identified.
    """

    NONE   = 0
    WHITE  = 1
    BLACK  = 2
    GRAY   = 3
    RED    = 4
    BLUE   = 5
    GREEN  = 6
    YELLOW = 7
    PURPLE = 8
    BROWN  = 9
    ORANGE = 10
