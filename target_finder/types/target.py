import shapes
import color
class Target:
    def __init__(self, x, y, shape):
        self.x = x
        self.y = y
        self.shape = shape

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def shape(self):
        return self._shape

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