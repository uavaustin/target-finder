"""Testing that the basic types in target_finder/types work."""

from target_finder import Color, Shape, Target


def test_basic_target():
    t = Target(0.5, 0.5, Shape.SQUARE)

    assert t.x == 0.5
    assert t.y == 0.5
    assert t.shape == Shape.SQUARE
