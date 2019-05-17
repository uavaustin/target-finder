"""Testing target finding."""

from target_finder.classification import find_targets
from target_finder import Color, Shape, Target

import PIL.Image
import os


IMAGE_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')


TESTS = [
    ('fake-1.jpg', [
        Target(x=502, y=2101, width=36, height=35, orientation=0.0,
               confidence=1,
               shape=Shape.PENTAGON, background_color=Color.BLACK,
               alphanumeric='C', alphanumeric_color=Color.RED),
        Target(x=701, y=551, width=50, height=50, orientation=0.0,
               confidence=1,
               shape=Shape.SQUARE, background_color=Color.GREEN,
               alphanumeric='E', alphanumeric_color=Color.WHITE),
        Target(x=849, y=1451, width=49, height=46, orientation=0.0,
               confidence=1,
               shape=Shape.CROSS, background_color=Color.BLUE,
               alphanumeric='E', alphanumeric_color=Color.WHITE),
        Target(x=1702, y=348, width=54, height=60, orientation=0.0,
               confidence=1,
               shape=Shape.SQUARE, background_color=Color.RED,
               alphanumeric='I', alphanumeric_color=Color.BROWN),
        Target(x=3201, y=648, width=52, height=64, orientation=0.0,
               confidence=1,
               shape=Shape.QUARTER_CIRCLE, background_color=Color.BLUE,
               alphanumeric='D', alphanumeric_color=Color.GREEN),
        Target(x=1250, y=1652, width=37, height=32, orientation=0.0,
               confidence=1,
               shape=Shape.CIRCLE, background_color=Color.BLUE,
               alphanumeric='R', alphanumeric_color=Color.BLACK),
        Target(x=3749, y=1747, width=61, height=56, orientation=0.0,
               confidence=1,
               shape=Shape.TRIANGLE, background_color=Color.BLACK,
               alphanumeric='unk', alphanumeric_color=Color.NONE),
        Target(x=1850, y=552, width=37, height=34, orientation=0.0,
               confidence=1,
               shape=Shape.QUARTER_CIRCLE, background_color=Color.BLACK,
               alphanumeric='E', alphanumeric_color=Color.YELLOW),
        Target(x=1203, y=651, width=51, height=51, orientation=0.0,
               confidence=1,
               shape=Shape.SEMICIRCLE, background_color=Color.BLUE,
               alphanumeric='T', alphanumeric_color=Color.WHITE),
        Target(x=2701, y=451, width=36, height=32, orientation=0.0,
               confidence=1,
               shape=Shape.PENTAGON, background_color=Color.PURPLE,
               alphanumeric='B', alphanumeric_color=Color.BLUE),
        Target(x=3551, y=651, width=37, height=33, orientation=0.0,
               confidence=1,
               shape=Shape.CROSS, background_color=Color.BLUE,
               alphanumeric='Q', alphanumeric_color=Color.GRAY),
        Target(x=2594, y=496, width=46, height=63, orientation=0.0,
               confidence=1,
               shape=Shape.TRIANGLE, background_color=Color.BLACK,
               alphanumeric='unk', alphanumeric_color=Color.NONE),
        Target(x=752, y=1097, width=44, height=46, orientation=0.0,
               confidence=1,
               shape=Shape.STAR, background_color=Color.YELLOW,
               alphanumeric='G', alphanumeric_color=Color.WHITE)
    ]),
    ('real-1.jpg', [
        Target(x=743, y=953, width=44, height=29, orientation=0.0,
               confidence=1,
               shape=Shape.SEMICIRCLE, background_color=Color.RED,
               alphanumeric='T', alphanumeric_color=Color.YELLOW),
        Target(x=1987, y=87, width=23, height=26, orientation=0.0,
               confidence=1,
               shape=Shape.TRIANGLE, background_color=Color.BLACK,
               alphanumeric='E', alphanumeric_color=Color.WHITE),
        Target(x=1956, y=910, width=28, height=32, orientation=0.0,
               confidence=1,
               shape=Shape.TRIANGLE, background_color=Color.ORANGE,
               alphanumeric='I', alphanumeric_color=Color.WHITE),
        Target(x=2450, y=1828, width=30, height=30, orientation=0.0,
               confidence=1,
               shape=Shape.TRAPEZOID, background_color=Color.WHITE,
               alphanumeric='S', alphanumeric_color=Color.BLACK),
        Target(x=2588, y=1554, width=21, height=20, orientation=0.0,
               confidence=1,
               shape=Shape.SQUARE, background_color=Color.BLUE,
               alphanumeric='unk', alphanumeric_color=Color.NONE)
    ]),
    ('real-2.jpg', [
        Target(x=1609, y=479, width=88, height=84, orientation=0.0,
               confidence=1,
               shape=Shape.TRIANGLE, background_color=Color.ORANGE,
               alphanumeric='I', alphanumeric_color=Color.WHITE)
    ])
]


def test_targets():

    for fn, expected_targets in TESTS:
        image = PIL.Image.open(os.path.join(IMAGE_DIR, fn))
        actual_targets = find_targets(image)
        _test_targets(actual_targets, expected_targets)


def _test_targets(actual_targets, expected_targets):

    found = 0
    correct_bbox = 0
    correct_colors = 0
    correct_shapes = 0
    correct_alphas = 0

    for expected_target in expected_targets:

        for actual_target in actual_targets:

            if expected_target.overlaps(actual_target):

                found += 1

                if _correct_bbox(expected_target, actual_target):
                    correct_bbox += 1

                if _correct_color(expected_target, actual_target):
                    correct_colors += 1

                if _correct_alpha(expected_target, actual_target):
                    correct_alphas += 1

                if _correct_shape(expected_target, actual_target):
                    correct_shapes += 1

                break

    # Fraction of targets found
    found_acc = found / len(actual_targets)

    # Fraction of targets correctly classified
    correct_cnts = [correct_bbox, correct_colors,
                    correct_shapes, correct_alphas]
    overall_acc = sum([v / len(actual_targets) for v in correct_cnts]) / 4

    print('Accuracy - Found: {}%, Classification: {}%'
          .format(found_acc * 100, overall_acc * 100))

    assert found_acc > 0.90
    assert overall_acc > 0.40  # set low due to poor color classification


def _correct_color(target_a, target_b):
    score = 0
    if target_a.background_color == target_b.background_color:
        score += 0.5
    if target_a.alphanumeric_color == target_b.alphanumeric_color:
        score += 0.5
    return score


def _correct_shape(target_a, target_b):
    return target_a.shape == target_b.shape


def _correct_alpha(target_a, target_b):
    return target_a.alphanumeric == target_b.alphanumeric


def _correct_bbox(target_a, target_b, max_error=20):
    diffs = [
        target_a.x - target_b.x,
        target_a.y - target_b.y,
        target_a.width - target_b.width,
        target_a.height - target_b.height
    ]
    return max([abs(v) for v in diffs]) <= max_error
