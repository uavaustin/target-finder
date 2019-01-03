"""Testing preprocessing in target_finder/preprocessing."""

import PIL.Image
import os

from target_finder.preprocessing import find_blobs
from target_finder.types import Blob

TESTS = [
    ('quarter_circle.jpg',
        Blob(x=11, y=13, width=60, height=59, has_mask=True)),
    ('semicircle.jpg',
        Blob(x=10, y=28, width=60, height=31, has_mask=True)),
    ('circle.jpg',
        Blob(x=10, y=10, width=61, height=62, has_mask=True)),
    ('triangle.jpg',
        Blob(x=12, y=15, width=58, height=52, has_mask=True)),
    ('trapezoid.jpg',
        Blob(x=11, y=19, width=57, height=45, has_mask=True)),
    ('square.jpg',
        Blob(x=17, y=18, width=47, height=47, has_mask=True)),
    ('rectangle.jpg',
        Blob(x=17, y=27, width=47, height=28, has_mask=True)),
    ('star.jpg',
        Blob(x=12, y=13, width=56, height=51, has_mask=True)),
    ('cross.jpg',
        Blob(x=17, y=17, width=47, height=47, has_mask=True)),
    ('pentagon.jpg',
        Blob(x=13, y=13, width=56, height=54, has_mask=True)),
    ('hexagon.jpg',
        Blob(x=10, y=14, width=60, height=54, has_mask=True)),
    ('heptagon.jpg',
        Blob(x=13, y=14, width=56, height=54, has_mask=True)),
    ('octagon.jpg',
        Blob(x=10, y=12, width=59, height=58, has_mask=True))
]

SHAPES_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')


def test_find_blobs():

    for name, expected_blob in TESTS:

        image = PIL.Image.open(os.path.join(SHAPES_DIR, name))
        blobs = find_blobs(image)

        assert len(blobs) > 0, f'{name} has no blobs'

        _assert_blobs_match(name, blobs[0], expected_blob)


def _assert_blobs_match(name, blob, expected_blob, thres=8):
    """See if these blobs are pretty much the same"""
    params = [
        blob.x - expected_blob.x,
        blob.y - expected_blob.y,
        blob.width - expected_blob.width,
        blob.height - expected_blob.height
    ]

    # check if any param is off by as much as `thres`
    diff = any(map(lambda d: abs(d) > thres, params))

    if diff:
        raise AssertionError(f'{name} does not match {expected_blob}')
