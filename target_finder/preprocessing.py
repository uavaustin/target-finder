"""Contains logic for finding and filtering blobs."""
import cv2
import numpy as np

from .types import BBox


def extract_crops(image, size, overlap):

    h, w, _ = image.shape

    crops = []

    for y1 in range(0, h - size[0], size[0] - overlap):

        for x1 in range(0, w - size[1], size[1] - overlap):

            y2 = y1 + size[0]
            x2 = x1 + size[1]

            crop_ary = image[y1:y2, x1:x2]
            box = BBox(x1, y1, x2, y2)
            box.image = crop_ary

            crops.append(box)

    return crops


def resize_all(image_crops, new_size):

    new_crops = []

    for crop in image_crops:

        new_image = cv2.resize(crop.image, new_size)
        box = BBox(crop.x1, crop.y1, crop.x2, crop.y2)
        box.image = new_image
        new_crops.append(box)

    return new_crops


def extract_contour(image):

    edges = cv2.Canny(image, 200, 500)
    ret, thresh = cv2.threshold(edges, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    main_contour = None
    max_area = 0
    for cnt in contours:
        x, y, width, height = cv2.boundingRect(cnt)
        if width * height > max_area:
            main_contour = cnt
            max_area = width * height

    return main_contour
