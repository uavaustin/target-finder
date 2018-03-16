"""Testing that the basic types in target_finder/types work."""

import PIL.Image
import pytest

from target_finder import Blob, Color, Shape, Target


def test_basic_blob():
    b = Blob(150, 250, 50, 100)

    assert b.x == 150
    assert b.y == 250
    assert b.width == 50
    assert b.height == 100
    assert b.image == None


def test_blob_with_image():
    image = PIL.Image.new('1', (200, 300))

    b = Blob(100, 200, 20, 30, image=image)

    assert b.x == 100
    assert b.y == 200
    assert b.width == 20
    assert b.height == 30
    assert b.image == image


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
    assert t.image == None
    assert t.confidence == 0.0


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
