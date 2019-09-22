"""Testing that the basic types in target_finder/types work."""

import PIL.Image
import pytest

from target_finder import Color, Shape, Target


def test_basic_target():
    t = Target(2, 4, 5, 8)

    assert t.x == 2
    assert t.y == 4
    assert t.width == 5
    assert t.height == 8
    assert t.shape == Shape.NAS
    assert t.orientation == 0.0
    assert t.background_color == Color.NONE
    assert t.alphanumeric == ''
    assert t.alphanumeric_color == Color.NONE
    assert t.image is None
    assert t.confidence == 0.0
    assert str(t) == "Target(x=2, y=4, width=5, height=8, " + \
                     "orientation=0.0, confidence=0.0, " + \
                     "shape=Shape.NAS, background_color=Color.NONE, " + \
                     "alphanumeric='', alphanumeric_color=Color.NONE)"
    assert repr(eval(repr(t))) == repr(t)


def test_target_with_optionals():
    image = PIL.Image.new('1', (200, 300))

    t = Target(3, 5, 7, 9, shape=Shape.SQUARE, orientation=74.3,
               background_color=Color.GREEN, alphanumeric='A',
               alphanumeric_color=Color.WHITE, image=image, confidence=0.97)

    assert t.x == 3
    assert t.y == 5
    assert t.width == 7
    assert t.height == 9
    assert t.shape == Shape.SQUARE
    assert t.orientation == 74.3
    assert t.background_color == Color.GREEN
    assert t.alphanumeric == 'A'
    assert t.alphanumeric_color == Color.WHITE
    assert t.image == image
    assert t.confidence == 0.97
    assert str(t) == "Target(x=3, y=5, width=7, height=9, " + \
                     "orientation=74.3, confidence=0.97, " + \
                     "shape=Shape.SQUARE, background_color=Color.GREEN, " + \
                     "alphanumeric='A', alphanumeric_color=Color.WHITE)"
    assert repr(eval(repr(t))) == repr(t)
