"""Contains logic for finding targets in blobs."""

import os
from pkg_resources import resource_filename

import cv2
import numpy as np
import os
import PIL.Image
import sklearn.cluster
import scipy.misc
import target_finder_model as tfm

from .darknet import Yolo3Detector, PreClassifier
from .preprocessing import extract_crops, resize_all, extract_contour
from .types import Color, Shape, Target, BBox


# Default Models w/default weights
models = {
    'yolo3': Yolo3Detector(),
    'clf': PreClassifier()
}


def set_models(new_models):
    models.update(new_models)


def find_targets(image, limit=20):

    image_ary = cv2.imread(image)

    raw_bboxes = _run_models(image_ary)
    targets = _bboxes_to_targets(raw_bboxes)

    # Sorting with highest confidence first.
    targets.sort(key=lambda t: t.confidence, reverse=True)
    _identify_colors(targets, image_ary)

    return targets[:limit]


def _run_models(image):

    detector_model = models['yolo3']
    clf_model = models['clf']

    crops = extract_crops(image, tfm.CROP_SIZE, tfm.CROP_OVERLAP)

    clf_crops = resize_all(crops, tfm.PRECLF_SIZE)
    regions = clf_model.classify_all([box.image for box in clf_crops])

    filtered_crops = [crops[i] for i, region in enumerate(regions)
                      if region == 'shape_target']

    detector_crops = resize_all(filtered_crops, tfm.DETECTOR_SIZE)

    offset_bboxes = detector_model.detect_all([box.image
                                               for box in detector_crops])

    ratio = tfm.DETECTOR_SIZE[0] / tfm.CROP_SIZE[0]
    normalized_bboxes = []

    for crop, bboxes in zip(detector_crops, offset_bboxes):
        for name, conf, bbox in bboxes:
            bw = bbox[2] / ratio
            bh = bbox[3] / ratio
            bx = (bbox[0] / ratio) + crop.x1
            by = (bbox[1] / ratio) + crop.y1
            box = BBox(bx, by, bx + bw, by + bh)
            box.meta = {name: conf}
            box.confidence = conf
            normalized_bboxes.append(box)

    return normalized_bboxes


def _bboxes_to_targets(bboxes):
    """Produce targets from bounding boxes"""

    targets = []
    merged_bboxes = _merge_boxes(bboxes)

    for box in merged_bboxes:
        shape, alpha, conf = _get_shape_and_alpha(box)
        targets.append(Target(box.x1, box.y1, box.w, box.h,
                              shape=shape,
                              alphanumeric=alpha,
                              confidence=conf))

    return targets


def _get_shape_and_alpha(box):

    best_shape, conf_shape = 'unk', 0
    best_alpha, conf_alpha = 'unk', 0

    for class_name, conf in box.meta.items():
        if len(class_name) == 1 and conf > conf_alpha:
            best_alpha = class_name
            conf_alpha = conf
        elif len(class_name) != 1 and conf > conf_shape:
            best_shape = class_name
            conf_shape = conf

    # convert name to object
    if best_shape == 'unk':
        shape = None
    else:
        shape = Shape[best_shape.upper().replace('-', '_')]

    return shape, best_alpha, ((conf_shape + conf_alpha) / 2)


def _merge_boxes(boxes):
    merged = []
    for box in boxes:
        for merged_box in merged:
            if _intersect(box, merged_box):
                _enlarge(merged_box, box)
                merged_box.meta.update(box.meta)
                break
        else:
            merged.append(box)
    return merged


def _intersect(box1, box2):
    # no intersection along x-axis
    if (box1.x1 > box2.x2 or box2.x1 > box1.x2):
        return False

    # no intersection along y-axis
    if (box1.y1 > box2.y2 or box2.y1 > box1.y2):
        return False

    return True


def _enlarge(main_box, new_box):
    main_box.x1 = min(main_box.x1, new_box.x1)
    main_box.x2 = max(main_box.x2, new_box.x2)
    main_box.y1 = min(main_box.y1, new_box.y1)
    main_box.y2 = max(main_box.y2, new_box.y2)


def _identify_colors(targets, full_image, padding=15):

    for target in targets:
        x = int(target.x) - padding
        y = int(target.y) - padding
        w = int(target.width) + padding * 2
        h = int(target.height) + padding * 2
        blob_image = full_image[y:y + h, x:x + w]
        try:
            target_color, alpha_color = _get_colors(blob_image)
            target.background_color = target_color
            target.alphanumeric_color = alpha_color
        except cv2.error:
            pass


def _get_colors(image):
    """Find the primary and seconday colors of the the blob"""

    contour = extract_contour(image)

    (color_a, count_a), (color_b, count_b) = _find_main_colors(image, contour)

    # this assumes the shape will have more pixels than alphanum
    if count_a > count_b:
        primary, secondary = color_a, color_b
    else:
        primary, secondary = color_b, color_a

    primary_color = _get_color_name(primary)
    secondary_color = _get_color_name(secondary)

    return primary_color, secondary_color


def _find_main_colors(image, contour):
    """Find the two main colors of the blob"""
    mask_img = np.array(image)  # the image w/the mask applied

    mask = np.zeros(mask_img.shape[:2], dtype='uint8')  # the mask itself

    # create mask
    cv2.drawContours(mask, [contour], -1, 255, -1)

    # apply mask
    masked_image = cv2.bitwise_and(mask_img, mask_img, mask=mask)

    # extract colors from region within mask
    mask_x, mask_y = np.nonzero(mask)
    valid_colors = masked_image[mask_x, mask_y].astype(np.float)

    # Get the two average colors
    algo = sklearn.cluster.AgglomerativeClustering(n_clusters=2)
    algo.fit(valid_colors)
    colors = algo.labels_
    all_a = valid_colors[colors == 1]
    all_b = valid_colors[colors == 0]

    # extract colors from prediction
    color_a = np.mean(all_a, axis=0)
    count_a = all_a.shape[0]
    color_b = np.mean(all_b, axis=0)
    count_b = all_b.shape[0]

    return (color_a, count_a), (color_b, count_b)


def _get_color_name(requested_color):
    # Calculates HSV values
    colors_set = (
        [25, Color.RED],
        [56, Color.ORANGE],
        [69, Color.YELLOW],
        [169, Color.GREEN],
        [274, Color.BLUE],
        [319, Color.PURPLE],
        [360, Color.RED]
    )

    r0 = requested_color[0]
    g0 = requested_color[1]
    b0 = requested_color[2]

    r = r0 / 255
    g = g0 / 255
    b = b0 / 255
    c_max = max(r, g, b)
    c_min = min(r, g, b)
    delta = c_max - c_min

    h = 0
    s = 0
    v = 0
    if delta == 0:
        h = 0
    elif c_max == r:
        h = 60 * (((g - b) / delta) % 6)
    elif c_max == g:
        h = 60 * (((b - r) / delta) + 2)
    elif c_max == b:
        h = 60 * (((r - g) / delta) + 4)

    if c_max == 0:
        s = 0
    else:
        s = delta / c_max

    v = c_max * 100
    s *= 100

    if 0 < v <= 25:
        return Color.BLACK
    elif 0 < s <= 20:
        if 25 < v < 80:
            return Color.GRAY
        else:
            return Color.WHITE

    for i in range(len(colors_set)):
        if h < colors_set[i][0]:
            if colors_set[i][1] == Color.ORANGE:
                if v < 60:
                    return Color.BROWN
                else:
                    return colors_set[i][1]
            else:
                return colors_set[i][1]

    return Color.NONE
