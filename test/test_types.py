"""Testing that the basic types in target_finder/types work."""

import pytest

from target_finder import Color, Shape, Target


def test_basic_target():
    t = Target(0.4, 0.6, Shape.SQUARE)

    assert t.x == 0.4
    assert t.y == 0.6
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


def test_target_orientation_range():
    t = Target(0.5, 0.5, Shape.CIRCLE)

    testing_values = [
        [1, 1],
        [359, 359],
        [0, 0],
        [360, 0],
        [-1, 359],
        [-1000, 80],
        [1000, 280],
        [123.456789, 123.456789]
    ]

    # Looping through them each and allowing for an absolute error of
    # 1e-10.
    for pair in testing_values:
        actual = pair[0]
        expected = pair[1]

        t.orientation = actual

        assert abs(expected - t.orientation) < 1e-10


def test_target_alphanumeric():
    t = Target(0.5, 0.5, Shape.OCTAGON)

    valid = ['A', 'Z', 'a', 'z', '0', '9', 'Word', '123', 'h4llo']
    invalid = [' ', 'Invalid Input', '#', '$$$', '*']

    for string in valid:
        t.alphanumeric = string
        assert t.alphanumeric == string

    for string in invalid:
        with pytest.raises(ValueError):
            t.alphanumeric = string
