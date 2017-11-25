"""Testing that the basic types in target_finder/types work."""

import pytest

from target_finder import Color, Shape, Target


def test_basic_target():
    t = Target(0.5, 0.5, Shape.SQUARE)

    assert t.x == 0.5
    assert t.y == 0.5
    assert t.shape == Shape.SQUARE


def test_target_x_y_min_range():
    with pytest.raises(ValueError):
        t_1 = Target(-0.01, 0.5, Shape.SQUARE)

    with pytest.raises(ValueError):
        t_2 = Target(0.5, -0.01, Shape.SQUARE)
        
    t_3 = Target(0, 0, Shape.SQUARE)

    with pytest.raises(ValueError):
        t_3.x = -0.01

    with pytest.raises(ValueError):
        t_3.y = -0.01

    assert t_3.x == 0
    assert t_3.y == 0


def test_target_x_y_max_range():
    with pytest.raises(ValueError):
        t_1 = Target(1.01, 1, Shape.SQUARE)

    with pytest.raises(ValueError):
        t_2 = Target(1, 1.01, Shape.SQUARE)
        
    t_3 = Target(1, 1, Shape.SQUARE)

    with pytest.raises(ValueError):
        t_3.x = 1.01

    with pytest.raises(ValueError):
        t_3.y = 1.01

    assert t_3.x == 1
    assert t_3.y == 1
