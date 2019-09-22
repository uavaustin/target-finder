"""Contains basic storage types passed around in the library."""

from enum import Enum, unique


@unique
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


@unique
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


class BBox(object):

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.image = None
        self.meta = ''
        self.confidence = -1

    @property
    def w(self):
        return self.x2 - self.x1

    @property
    def h(self):
        return self.y2 - self.y1


class Target(object):
    """Represents a target found on an image.

    This is intended to be built upon as the the target is being
    classified.

    Note that a target must have at least an x-position, y-position,
    width, and height to be created.

    A target should only be returned back by the library if at also
    contains a background color as well.

    Attributes:
        x (int): The x position of the top-left corner in pixels.
        y (int): The y position of the top-left corner in pixels.
        width (int): The width of the blob in pixels.
        height (int): The height of the blob in pixels.
        orientation (float): The orientation of the target. An
            orientation of 0 means the target is not rotated, an
            orientation of 90 means it the top of the target points
            to the right of the image (0 <= orientation < 360).
        shape (Shape): The target shape.
        background_color (Color): The target background color.
        alphanumeric (str): The letter(s) and/or number(s) on the
            target. May consist of one or more of the characters 0-9,
            A-Z, a-z. Typically, this will only be one capital
            letter.
        alphanumeric_color (Color): The target alphanumeric color.
        image (PIL.Image): Image showing the target.
        confidence (float): The confidence that the target exists
            (0 <= confidence <= 1).
    """

    def __init__(self, x, y, width, height, shape=Shape.NAS, orientation=0.0,
                 background_color=Color.NONE, alphanumeric='',
                 alphanumeric_color=Color.NONE, image=None, confidence=0.0):
        """Create a new Target object."""
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.shape = shape
        self.orientation = orientation
        self.background_color = background_color
        self.alphanumeric = alphanumeric
        self.alphanumeric_color = alphanumeric_color
        self.image = image
        self.confidence = confidence

    def overlaps(self, other_target):

        if (self.x > other_target.x + other_target.width or
           other_target.x > self.x + self.width):
            return False

        if (self.y > other_target.y + other_target.height or
           other_target.y > self.y + self.height):
            return False

        return True

    def __repr__(self):
        return str(self)

    def __str__(self):
        return (
            f'Target(x={self.x}, y={self.y}, width={self.width}, '
            f'height={self.height}, orientation={self.orientation}, '
            f'confidence={round(self.confidence, 2)}, shape={self.shape}, '
            f'background_color={self.background_color}, '
            f'alphanumeric={repr(self.alphanumeric)}, '
            f'alphanumeric_color={self.alphanumeric_color})'
        )
