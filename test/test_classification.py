"""Testing preprocessing in target_finder/preprocessing."""

import PIL.Image
import os

from target_finder.classification import find_targets
from target_finder.types import Target, Shape, Color

REQUIRED_CORRECT = 0.35  # accuracy required to pass build

TESTS = [
    ('quarter_circle.jpg',
        Target(x=0, y=0, width=80, height=80,
               orientation=0.0, confidence=1,
               shape=Shape.QUARTER_CIRCLE, background_color=Color.YELLOW,
               alphanumeric='K', alphanumeric_color=Color.PURPLE)),
    ('semicircle.jpg',
        Target(x=10, y=28, width=60, height=31,
               orientation=0.0, confidence=1,
               shape=Shape.SEMICIRCLE, background_color=Color.ORANGE,
               alphanumeric='U', alphanumeric_color=Color.GREEN)),
    ('circle.jpg',
        Target(x=10, y=10, width=61, height=62,
               orientation=0.0, confidence=1,
               shape=Shape.CIRCLE, background_color=Color.RED,
               alphanumeric='K', alphanumeric_color=Color.BLUE)),
    ('triangle.jpg',
        Target(x=12, y=15, width=58, height=52,
               orientation=0.0, confidence=1,
               shape=Shape.TRIANGLE, background_color=Color.RED,
               alphanumeric='K', alphanumeric_color=Color.GRAY)),
    ('trapezoid.jpg',
        Target(x=11, y=19, width=57, height=45,
               orientation=0.0, confidence=1,
               shape=Shape.TRAPEZOID, background_color=Color.GREEN,
               alphanumeric='R', alphanumeric_color=Color.RED)),
    ('square.jpg',
        Target(x=17, y=18, width=47, height=47,
               orientation=0.0, confidence=1,
               shape=Shape.SQUARE, background_color=Color.BLUE,
               alphanumeric='U', alphanumeric_color=Color.GREEN)),
    ('rectangle.jpg',
        Target(x=17, y=27, width=47, height=28,
               orientation=0.0, confidence=1,
               shape=Shape.RECTANGLE, background_color=Color.GREEN,
               alphanumeric='L', alphanumeric_color=Color.PURPLE)),
    ('star.jpg',
        Target(x=12, y=13, width=56, height=51,
               orientation=0.0, confidence=1,
               shape=Shape.STAR, background_color=Color.ORANGE,
               alphanumeric='R', alphanumeric_color=Color.RED)),
    ('cross.jpg',
        Target(x=17, y=17, width=47, height=47,
               orientation=0.0, confidence=1,
               shape=Shape.CROSS, background_color=Color.RED,
               alphanumeric='L', alphanumeric_color=Color.PURPLE)),
    ('pentagon.jpg',
        Target(x=13, y=13, width=56, height=54,
               orientation=0.0, confidence=1,
               shape=Shape.PENTAGON, background_color=Color.BLUE,
               alphanumeric='R', alphanumeric_color=Color.ORANGE)),
    ('hexagon.jpg',
        Target(x=10, y=14, width=60, height=54,
               orientation=0.0, confidence=1,
               shape=Shape.HEXAGON, background_color=Color.WHITE,
               alphanumeric='U', alphanumeric_color=Color.RED)),
    ('heptagon.jpg',
        Target(x=13, y=14, width=56, height=54,
               orientation=0.0, confidence=1,
               shape=Shape.HEPTAGON, background_color=Color.PURPLE,
               alphanumeric='K', alphanumeric_color=Color.GREEN)),
    ('octagon.jpg',
        Target(x=10, y=12, width=59, height=58,
               orientation=0.0, confidence=1,
               shape=Shape.OCTAGON, background_color=Color.BLUE,
               alphanumeric='U', alphanumeric_color=Color.ORANGE))
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

    # check accuracy before passing/failing builds
    accuracy = 1 - len(wrong) / len(TESTS)

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

    features = [
        target.shape == expected_target.shape,
        target.background_color == expected_target.background_color,
        # target.alphanumeric == expected_target.alphanumeric,
        target.alphanumeric_color == expected_target.alphanumeric_color
    ]

    # check if any param is off by as much as `thres`
    diff_params = any(map(lambda d: abs(d) > thres, params))

    # check if any features diff
    diff_features = not all(features)

    return diff_params or diff_features
