"""Contains logic for finding and filtering blobs."""
import cv2
import numpy as np

from .types import BBox


def extract_crops(image, size, overlap):

    h, w, _ = image.shape

    crops = []

    for y1 in range(0, h, size[0] - overlap):

        for x1 in range(0, w, size[1] - overlap):

            if y1 + size[0] > h:
                y1 = h - size[0]

            if x1 + size[1] > w:
                x1 = w - size[1]

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


def extract_contour(img):

    h, w, _ = img.shape

    # Seperate foreground w/YOLO's bbox as reference
    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    bbox = (24, 24, w - 52, h - 52)
    cv2.grabCut(img, mask, bbox, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Extract shape using foreground mask
    mask_fg = np.where((mask == 2) | (mask == 0), 0, 1)
    img_fg = (img * mask_fg[:, :, np.newaxis]).astype('uint8')

    # Create binary contour
    ret, thresh = cv2.threshold(img_fg, 10, 255, 0)
    contours, _ = cv2.findContours(thresh[:, :, 0], cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    main_contour = None
    max_area = 0
    for cnt in contours:
        x, y, width, height = cv2.boundingRect(cnt)
        if width * height > max_area:
            main_contour = cnt
            max_area = width * height

    return main_contour
