"""Contains logic for finding targets in blobs."""
from pkg_resources import resource_filename

import numpy as np
import PIL.Image
from PIL import ImageFilter
from sklearn.cluster import KMeans
import scipy.misc
import scipy.cluster
import heapq
import target_finder_model as tfm

from target_finder import preprocessing
from target_finder import types
from target_finder import color_cube

# Default Models w/default weights
models = {
    "frcnn": tfm.inference.DetectionModel(),
    "clf": tfm.inference.ClfModel(),
}

crop_size = (
    tfm.CONFIG["inputs"]["cropping"]["width"],
    tfm.CONFIG["inputs"]["cropping"]["height"],
)

overlap = tfm.CONFIG["inputs"]["cropping"]["overlap"]

pre_clf_size = (
    tfm.CONFIG["inputs"]["preclf"]["width"],
    tfm.CONFIG["inputs"]["preclf"]["height"],
)

det_size = (
    tfm.CONFIG["inputs"]["detector"]["width"],
    tfm.CONFIG["inputs"]["detector"]["height"],
)


def load_models():
    models["frcnn"].load()
    models["clf"].load()


def find_targets(pil_image, **kwargs):
    """Wrapper for finding targets which accepts a PIL image"""
    return find_targets_from_array(pil_image, **kwargs)


def find_targets_from_array(image_ary, limit=20):

    raw_bboxes = _run_models(image_ary)
    targets = _bboxes_to_targets(raw_bboxes)

    # Sorting with highest confidence first.
    targets.sort(key=lambda t: t.confidence, reverse=True)
    _identify_properties(targets, image_ary)

    return targets[:limit]


def _run_models(image):

    detector_model = models["frcnn"]
    clf_model = models["clf"]

    crops = preprocessing.extract_crops(image, crop_size, overlap)
    clf_crops = preprocessing.resize_all(crops, pre_clf_size)

    regions = clf_model.predict([box.image for box in clf_crops])

    filtered_crops = [
        crops[i] for i, region in enumerate(regions) if region.class_idx == 1
    ]

    # TODO(alex) Determine if this Sharpening is useful
    """
    for idx, crop in enumerate(filtered_crops):
        crop.image = crop.image.filter(ImageFilter.SHARPEN)
    """
    detector_crops = preprocessing.resize_all(filtered_crops, det_size)

    if len(detector_crops) != 0:
        offset_dets = detector_model.predict([box.image for box in detector_crops])
    else:
        offset_dets = []

    ratio = det_size[0] / crop_size[0]
    normalized_bboxes = []

    for crop, offset_dets in zip(detector_crops, offset_dets):

        for det in offset_dets:

            bw = det.width / ratio
            bh = det.height / ratio
            bx = (det.x / ratio) + crop.x1
            by = (det.y / ratio) + crop.y1
            box = types.BBox(bx, by, bx + bw, by + bh)
            box.meta = {det.class_name: det.confidence}
            box.confidence = det.confidence
            normalized_bboxes.append(box)

    return normalized_bboxes


def _bboxes_to_targets(bboxes):
    """Produce targets from bounding boxes"""

    targets = []
    merged_bboxes = _merge_boxes(bboxes)

    for box in merged_bboxes:
        shape, alpha, conf = _get_shape_and_alpha(box)
        targets.append(
            types.Target(
                box.x1,
                box.y1,
                box.w,
                box.h,
                shape=shape,
                alphanumeric=alpha,
                confidence=conf,
            )
        )

    return targets


def _get_shape_and_alpha(box):

    best_shape, conf_shape = "unk", 0
    best_alpha, conf_alpha = "unk", 0

    for class_name, conf in box.meta.items():
        if len(class_name) == 1 and conf > conf_alpha:
            best_alpha = class_name
            conf_alpha = conf
        elif len(class_name) != 1 and conf > conf_shape:
            best_shape = class_name
            conf_shape = conf

    # convert name to object
    if best_shape == "unk":
        shape = types.Shape.NAS
    else:
        shape = types.Shape[best_shape.upper().replace("-", "_")]

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
    if box1.x1 > box2.x2 or box2.x1 > box1.x2:
        return False

    # no intersection along y-axis
    if box1.y1 > box2.y2 or box2.y1 > box1.y2:
        return False

    return True


def _enlarge(main_box, new_box):
    main_box.x1 = min(main_box.x1, new_box.x1)
    main_box.x2 = max(main_box.x2, new_box.x2)
    main_box.y1 = min(main_box.y1, new_box.y1)
    main_box.y2 = max(main_box.y2, new_box.y2)


def _identify_properties(targets, full_image, padding=15):

    for target in targets:

        x = int(target.x) - padding
        y = int(target.y) - padding
        w = int(target.width) + padding * 2
        h = int(target.height) + padding * 2
        blob_image = full_image.crop((x, y, x + w, y + h))

        target.image = full_image

        try:
            target_color, alpha_color = _get_colors(blob_image)
            target.background_color = target_color
            target.alphanumeric_color = alpha_color
        except Exception as e:
            target.background_color = types.Color.NONE
            target.alphanumeric_color = types.Color.NONE


def _get_colors(image):
    """Find the primary and seconday colors of the the blob"""

    (color_a, count_a), (color_b, count_b) = _find_main_colors(image)

    # this assumes the shape will have more pixels than alphanum
    if count_a > count_b:
        primary, secondary = color_a, color_b
    else:
        primary, secondary = color_b, color_a

    primary_color = _get_color_name(primary)
    secondary_color = _get_color_name(secondary)

    return primary_color, secondary_color


def _find_main_colors(image):
    """Find the two main colors of the blob"""
    # TODO see: https://github.com/uavaustin/target-finder/issues/16
    ar = np.asarray(image)
    shape = ar.shape
    ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

    codes, dist = scipy.cluster.vq.kmeans(ar, 3)

    vecs, dist = scipy.cluster.vq.vq(ar, codes)  # assign codes
    counts, bins = scipy.histogram(vecs, len(codes))  # count occurrences
    top2 = heapq.nlargest(3, counts)  # find most frequent

    color_a = codes[np.where(counts == top2[0])][0]
    color_a = (color_a[0], color_a[1], color_a[2])
    count_a = top2[0]

    color_b = codes[np.where(counts == top2[1])][0]
    color_b = (color_b[0], color_b[1], color_b[2])
    count_b = top2[1]

    return (color_a, count_a), (color_b, count_b)


def _get_color_name(requested_color):

    # ColorCube((Hl, sl, vl), (Hu, Su, Vu))
    color_cubes = {
        "white": color_cube.ColorCube((0, 0, 85), (359, 20, 100)),
        "black": color_cube.ColorCube((0, 0, 0), (359, 100, 25)),
        "gray": color_cube.ColorCube((0, 0, 25), (359, 5, 75)),
        "blue": color_cube.ColorCube((180, 70, 70), (345, 100, 100)),
        "red": color_cube.ColorCube((350, 70, 70), (359, 100, 65)),
        "green": color_cube.ColorCube((100, 60, 30), (160, 100, 100)),
        "yellow": color_cube.ColorCube((60, 50, 55), (75, 100, 100)),
        "purple": color_cube.ColorCube((230, 40, 55), (280, 100, 100)),
        "brown": color_cube.ColorCube((300, 38, 20), (359, 100, 40)),
        "orange": color_cube.ColorCube((15, 70, 75), (45, 100, 100)),
    }

    r = requested_color[0] / 255
    g = requested_color[1] / 255
    b = requested_color[2] / 255

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

    contains = [color_cubes[key].contains((h, s, v)) for key in color_cubes]
    if not any(contains):
        cl_dists = [
            color_cubes[key].get_closest_distance((h, s, v)) for key in color_cubes
        ]
        dist = [
            np.sqrt((p[0] * p[0]) + (p[1] * p[1]) + (p[2] * p[2])) for p in cl_dists
        ]
        index = dist.index(min(dist)) + 1
        return types.Color(index)
    else:
        index = contains.index(True) + 1
        return types.Color(index)

    return types.Color.NONE
