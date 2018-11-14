"""Testing preprocessing in target_finder/preprocessing."""

import PIL.Image
import os

from target_finder.classification import find_targets
from target_finder.types import Target, Shape, Color

REQUIRED_CORRECT = 0.60 # accuracy required to pass build

TESTS = [
    ('quarter_circle.jpg',
        Target(x=0, y=0, width=80, height=80,
               orientation=0.0, confidence=1,
               shape=Shape.QUARTER_CIRCLE, background_color=Color.GRAY,
               alphanumeric='', alphanumeric_color=Color.YELLOW)),
    ('semicircle.jpg',
        Target(x=10, y=28, width=60, height=31,
               orientation=0.0, confidence=1,
               shape=Shape.SEMICIRCLE, background_color=Color.ORANGE,
               alphanumeric='', alphanumeric_color=Color.GREEN)),
    ('circle.jpg',
        Target(x=10, y=10, width=61, height=62,
               orientation=0.0, confidence=1,
               shape=Shape.CIRCLE, background_color=Color.RED,
               alphanumeric='', alphanumeric_color=Color.BLUE)),
    ('triangle.jpg',
        Target(x=12, y=15, width=58, height=52,
               orientation=0.0, confidence=1,
               shape=Shape.TRIANGLE, background_color=Color.RED,
               alphanumeric='', alphanumeric_color=Color.GRAY)),
    ('trapezoid.jpg',
        Target(x=11, y=19, width=57, height=45,
               orientation=0.0, confidence=1,
               shape=Shape.TRAPEZOID, background_color=Color.GREEN,
               alphanumeric='', alphanumeric_color=Color.RED)),
    ('square.jpg',
        Target(x=17, y=18, width=47, height=47,
               orientation=0.0, confidence=1,
               shape=Shape.SQUARE, background_color=Color.BLUE,
               alphanumeric='', alphanumeric_color=Color.GREEN)),
    ('star.jpg',
        Target(x=12, y=13, width=56, height=51,
               orientation=0.0, confidence=1,
               shape=Shape.STAR, background_color=Color.ORANGE,
               alphanumeric='', alphanumeric_color=Color.RED)),
    ('pentagon.jpg',
        Target(x=13, y=13, width=56, height=54,
               orientation=0.0, confidence=1,
               shape=Shape.PENTAGON, background_color=Color.BLUE,
               alphanumeric='', alphanumeric_color=Color.ORANGE)),
    ('hexagon.jpg',
        Target(x=10, y=14, width=60, height=54,
               orientation=0.0, confidence=1,
               shape=Shape.HEXAGON, background_color=Color.WHITE,
               alphanumeric='', alphanumeric_color=Color.RED)),
    ('heptagon.jpg',
        Target(x=13, y=14, width=56, height=54,
               orientation=0.0, confidence=1,
               shape=Shape.HEPTAGON, background_color=Color.PURPLE,
               alphanumeric='', alphanumeric_color=Color.GREEN)),
    ('octagon.jpg',
        Target(x=10, y=12, width=59, height=58,
               orientation=0.0, confidence=0.56,
               shape=Shape.OCTAGON, background_color=Color.BLUE,
               alphanumeric='', alphanumeric_color=Color.RED))
]

SHAPES_DIR = os.path.join(os.path.dirname(__file__), 'perfect_shapes')


def test_find_blobs():

    wrong = []

    for name, expected_target in TESTS:

        image = PIL.Image.open(os.path.join(SHAPES_DIR, name))
        targets = find_targets(image, min_confidence=0.40)

        assert len(targets) > 0, f'{name} has no targets'

        if _targets_diff(name, targets[0], expected_target):
            wrong.append((name, expected_target))

    accuracy = 1 - (len(wrong) / len(TESTS))

    if accuracy < REQUIRED_CORRECT:
        assert_msg = f'Targets {accuracy * 100}% correct.'
        for name, expected_target in wrong:
            assert_msg += f'\n{name} does not match {expected_target}.'
        raise AssertionError(assert_msg)


def _targets_diff(name, target, expected_target, thres=8):
    """See if these blobs are pretty much the same"""
    params = [
         target.x - expected_target.x,
         target.y - expected_target.y,
         target.width - expected_target.width,
         target.height - expected_target.height
    ]

    # check if any param is off by as much as `thres`
    diff = any(map(lambda d: abs(d) > thres, params))

    return diff
