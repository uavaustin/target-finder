"""Contains the Target class."""

from math import fmod

from .color import Color
from .shape import Shape


class Target(object):
    """Represets a target found on an image.

    This is intended to be built upon as the the target is being
    classified.

    Note that a target must have at least an x-position, y-position,
    and a shape to be created.

    A target should only be returned back by the library if at also
    contains a background color as well.

    Attributes:
        x (float): The x position from left to right (0 <= x <= 1).
        y (float): The y position from bottom to top (0 <= y <= 1).
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
    """

    def __init__(self, x, y, shape):
        """Create a new Target object.

        Args:
            x (float): The x position from left to right
                (0 <= x <= 1).
            y (float): The y position from bottom to top
                (0 <= y <= 1).
            shape (Shape): The target shape.
        """

        self.x = x
        self.y = y
        self.shape = shape

        self.orientation = None
        self.background_color = Color.NONE
        self.alphanumeric = None
        self.alphanumeric_color = Color.NONE

    @property
    def x(self):
        """The x position from left to right."""
        return self._x

    @x.setter
    def x(self, x):
        if (x >= 0 and x <= 1):
            self._x = x
        else:
            raise ValueError('0 <= x <= 1')

    @property
    def y(self):
        """"The y position from bottom to top."""
        return self._y

    @y.setter
    def y(self, y):
        if y >= 0 and y <= 1:
               self._y = y
        else:
            raise ValueError('0 <= y <= 1')

    @property
    def orientation(self):
        """The orientation of the target."""
        return self._orientation

    @orientation.setter
    def orientation(self, orientation):
        if orientation is None:
            self._orientation = None
        else:
            # Make sure the reading is in [0, 360). Note that fmod()
            # is better with floats than the mod operator %.
            self._orientation = fmod(fmod(orientation, 360) + 360, 360)

    @property
    def shape(self):
        """The target shape."""
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape

    @property
    def background_color(self):
        """The target background color."""
        return self._background_color

    @background_color.setter
    def background_color(self, color):
        self._background_color = color

    @property
    def alphanumeric(self):
        """The letter(s) and/or number(s) on the target."""
        return self._alphanumeric

    @alphanumeric.setter
    def alphanumeric(self, alphanumeric):
        if alphanumeric is None:
            self._alphanumeric = None
        elif alphanumeric.isalnum():
            self._alphanumeric = alphanumeric
        else:
            raise ValueError('Must be alphanumeric')

    @property
    def alphanumeric_color(self):
        """The target alphanumeric color."""
        return self._alphanumeric_color

    @alphanumeric_color.setter
    def alphanumeric_color(self, color):
        self._alphanumeric_color = color
