"""Contains logic for finding and filtering blobs."""
import numpy as np

from .types import BBox


def extract_crops(image, size, overlap):

    w, h = image.size

    crops = []

    for y1 in range(0, h, size[0] - overlap):

        for x1 in range(0, w, size[1] - overlap):

            if y1 + size[0] > h:
                y1 = h - size[0]

            if x1 + size[1] > w:
                x1 = w - size[1]

            y2 = y1 + size[0]
            x2 = x1 + size[1]

            box = BBox(x1, y1, x2, y2)
            box.image = image.crop((x1, y1, x2, y2))

            crops.append(box)

    return crops


def resize_all(image_crops, new_size):

    new_crops = []

    for crop in image_crops:

        new_image = crop.image.resize(new_size)
        box = BBox(crop.x1, crop.y1, crop.x2, crop.y2)
        box.image = new_image
        new_crops.append(box)

    return new_crops
