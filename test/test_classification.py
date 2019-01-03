"""Testing classification in target_finder/classification."""

import os
import warnings

import PIL.Image
import pytest

from target_finder.classification import find_targets
from target_finder.types import Target, Shape, Color


# Required accuracies for localization, shape, and color detection.
# Note that color accuracy is separates background and alpha, i.e.
# just missing one color just takes off half.
REQ_LOC_ACCURACY = 1.00
REQ_SHAPE_ACCURACY = 0.46
REQ_COLOR_ACCURACY = 0.84

# Localization tolerance for x, y, width, and height.
LOC_TOL = 8

TESTS = [
    ('quarter_circle.jpg',
        Target(x=11, y=13, width=59, height=58,
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

SHAPES_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')


# All the found targets are done before the assertions. Only run
# once if successful.
@pytest.fixture(scope='module')
def found_targets():
    found = []

    for name, expected in TESTS:
        image = PIL.Image.open(os.path.join(SHAPES_DIR, name))

        # FIXME: We shouldn't have to set the confidence manually
        #        here. The default should be used in tests in the
        #        future.
        targets = find_targets(image, min_confidence=0.40)

        found.append((name, targets, expected))

    return found


# All should have one target attached.
def test_existence(found_targets):
    for name, targets, expected in found_targets:
        assert len(targets) > 0, f'{name} does not have only 1 target'


def test_localization(found_targets):
    diffs = _diff_all(found_targets, _diff_localization)
    accuracy = 1 - len(diffs) / len(found_targets)

    _message_if_needed('Localization', diffs, accuracy, REQ_LOC_ACCURACY)


def test_shapes(found_targets):
    diffs = _diff_all(found_targets, _diff_shape)
    accuracy = 1 - len(diffs) / len(found_targets)

    _message_if_needed('Shapes', diffs, accuracy, REQ_SHAPE_ACCURACY)


def test_colors(found_targets):
    background_diffs = _diff_all(found_targets, _diff_background_color)
    alphanumeric_diffs = _diff_all(found_targets, _diff_alphanumeric_color)

    # Group the same shapes together.
    diffs = sorted(background_diffs + alphanumeric_diffs)

    accuracy = 1 - len(diffs) / (2 * len(found_targets))

    _message_if_needed('Colors', diffs, accuracy, REQ_COLOR_ACCURACY)


def _diff_all(found_targets, diff_fn):
    diffs = []

    for name, targets, expected in found_targets:
        diff = diff_fn(targets[0], expected)

        if diff:
            diffs.append(f'{name}: {diff}')

    return diffs


def _diff_localization(target, expected_target):
    params = [
        target.x - expected_target.x,
        target.y - expected_target.y,
        target.width - expected_target.width,
        target.height - expected_target.height
    ]

    # Check if any are too far off.
    diff_params = any(map(lambda d: abs(d) > LOC_TOL, params))

    if diff_params:
        return 'localization is not within tolerance'


def _diff_shape(target, expected_target):
    actual = target.shape
    expected = expected_target.shape

    if actual != expected:
        return f'shape {actual} does not match {expected}'


def _diff_background_color(target, expected_target):
    actual = target.background_color
    expected = expected_target.background_color

    if actual != expected:
        return f'background color {actual} does not match {expected}'


def _diff_alphanumeric_color(target, expected_target):
    actual = target.alphanumeric_color
    expected = expected_target.alphanumeric_color

    if actual != expected:
        return f'alphanumeric color {actual} does not match {expected}'


def _message_if_needed(attr, diffs, accuracy, req_accuracy):
    # Nothing to message if there are no diffs.
    if not diffs:
        return

    message = f'{attr} {round(accuracy * 100, 2)}% correct.'

    for diff in diffs:
        message += '\n' + diff

    if accuracy < req_accuracy:
        raise AssertionError(message)
    else:
        message += f'\nThis still meets the {req_accuracy} threshold.'
        warnings.warn(message)
