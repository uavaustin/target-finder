"""Contains the Blob class."""

class Blob(object):
    """Represents a block with a possible target.

    This is for the middle stage before the target is identified.

    This may be returned back by the library if only blobs are being
    requested. This could be so that the logic is being seperated out
    externally.

    Attributes:
        x (float): The x position from left to right (0 <= x <= 1).
        y (float): The y position from bottom to top (0 <= y <= 1).
        image (PIL.Image): Image for the blob.
    """

    def __init__(self, x, y, image):
        """Create a new Target object.

        Args:
            x (float): The x position from left to right
                (0 <= x <= 1).
            y (float): The y position from bottom to top
                (0 <= y <= 1).
        """

        self.x = x
        self.y = y
        self.image = image

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
    def image(self):
        """Image for the blob."""
        return self._image

    @image.setter
    def image(self, image):
        self._image = image
