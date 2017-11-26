"""Contains the Shape enum for identifying target shapes."""

from enum import Enum


class Shape(Enum):
    """Contains target shapes for the AUVSI SUAS Interop Server.

    Note that Shape.NAS (not-a-shape) can be used if a shape cannot
    be identified.
    """

    NAS            = 0
    CIRCLE         = 1
    SEMICIRCLE     = 2
    QUARTER_CIRCLE = 3
    TRIANGLE       = 4
    SQUARE         = 5
    RECTANGLE      = 6
    TRAPEZOID      = 7
    PENTAGON       = 8
    HEXAGON        = 9
    HEPTAGON       = 10
    OCTAGON        = 11
    STAR           = 12
    CROSS          = 13
