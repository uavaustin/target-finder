import shapes
import color
import string
import numpy as np
class Target:
    X = np.linspace(0,1,101)
    Y = np.linspace(0,1,101)
    ORIENTATION = list(range(0,360))
    SHAPE = shapes
    BACKGROUND_COLOR = color
    ALPHANUMERIC = list(string.ascii_uppercase) + list(range(1,11))
    ALPHANUMERIC_COLOR = color