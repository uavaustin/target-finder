"""Testing colors from target_finder/classification."""

import os
import warnings

import PIL.Image

from target_finder.classification import _get_color, _get_color_name
from target_finder.classification import find_blobs
from target_finder.types import Color

REQUIRED_CORRECT = 0.8  # accuracy required to pass build

TESTS = [
    ('quarter_circle.jpg', Color.YELLOW, Color.PURPLE),
    ('semicircle.jpg', Color.ORANGE, Color.GREEN),
    ('circle.jpg', Color.RED, Color.BLUE),
    ('triangle.jpg', Color.RED, Color.GRAY),
    ('trapezoid.jpg', Color.BLACK, Color.RED),
    ('square.jpg', Color.BLUE, Color.GREEN),
    ('rectangle.jpg', Color.GREEN, Color.PURPLE),
    ('star.jpg', Color.ORANGE, Color.RED),
    ('cross.jpg', Color.RED, Color.PURPLE),
    ('pentagon.jpg', Color.BLUE, Color.ORANGE),
    ('hexagon.jpg', Color.WHITE, Color.RED),
    ('heptagon.jpg', Color.PURPLE, Color.GREEN),
    ('octagon.jpg', Color.BLUE, Color.RED)
]

SHAPES_DIR = os.path.join(os.path.dirname(__file__), 'perfect_shapes')


def test_colors():

    wrong = []

    for name, expected_bg, expected_alpha in TESTS:

        image = PIL.Image.open(os.path.join(SHAPES_DIR, name))

        blobs = find_blobs(image)

        assert len(blobs) > 0, f'{name} should have blobs, but none found.'

        primary_rgb, secondary_rgb = _get_color(blobs[0])
        primary = _get_color_name(primary_rgb, None)
        secondary = _get_color_name(secondary_rgb, primary)

        if primary != expected_bg:
            wrong.append((name, primary, expected_bg))

        if secondary != expected_alpha:
            wrong.append((name, secondary, expected_alpha))

    # check accuracy before passing/failing builds
    accuracy = 1 - len(wrong) / len(TESTS)

    if len(wrong):
        message = f'Colors {round(accuracy * 100, 2)}% correct.'
        for name, actual, expected in wrong:
            message += f'\n{actual} does not match {expected} for {name}.'

        if accuracy < REQUIRED_CORRECT:
            raise AssertionError(message)
        else:
            message += f'\nThis still meets the {REQUIRED_CORRECT} threshold.'
            warnings.warn(message)
