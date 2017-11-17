import shapes
import color
class Target:
    def __init__(self, x, y, shape, background_color, alphanumeric_color = None, orientation = None, alphanumeric = None):
        self.x = x
        self.y = y
        self.shape = shape
        self.background_color = color

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def shape(self):
        return self._shape

    @property
    def background_color(self):
        return self._background_color

    @property
    def alphanumeric_color(self):
        return self._alphanumeric_color

    @property
    def orientation(self):
        return self._orientation

    @property
    def alphanumeric(self):
        return self._alphanumeric

    @x.setter
    def x(self, x):
        if (x >= 0 and x <= 1):
            self._x = x
        else:
            raise ValueError('0 <= x <= 1')

    @y.setter
    def y(self, y):
        if y >= 0 and y <= 1:
               self._y = y
        else:
            raise ValueError('0 <= y <= 1')

    @shape.setter
    def shape(self, shapes):
        self._shape = shapes

    @background_color.setter
    def background_color(self, color):
        self._background_color = color

    @alphanumeric_color.setter
    def alphanumeric_color(self, color):
        return self._alphanumeric_color

    @orientation.setter
    def orientation(self, orientation):
        if orientation >= 360:
            orientation = orientation % 360
            self._orientation = orientation
        else:
            self._orientation = orientation

    @alphanumeric.setter
    def alphanumeric(self, alphanumeric):
        self._alphanumeric
        if isalphnum(alphanumeric):
            self._alphanumeric = alphanumeric
        else:
            raise ValueError('Must be an alphanumeric')
