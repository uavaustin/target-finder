"""Contains logic for finding targets in blobs."""

import os
import sys

from pkg_resources import resource_filename
import cv2
import numpy as np
import PIL.Image
import sklearn.cluster
import scipy.misc
import target_finder_model as tfm

from datetime import datetime
from target_finder.darknet import Yolo3Detector, PreClassifier
from target_finder.preprocessing import extract_crops, resize_all, extract_contour
from target_finder.types import Color, Shape, Target, BBox
from target_finder.color_cube import ColorCube

# Default Models w/default weights
models = {
    'yolo3': Yolo3Detector(),
    'clf': PreClassifier()
}


def set_models(new_models):
    models.update(new_models)


def find_targets(pil_image, **kwargs):
    """Wrapper for finding targets which accepts a PIL image"""
    image_ary = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return find_targets_from_array(image_ary, **kwargs)


def find_targets_from_array(image_ary, limit=20):

    raw_bboxes = _run_models(image_ary)
    targets = _bboxes_to_targets(raw_bboxes)

    # Sorting with highest confidence first.
    targets.sort(key=lambda t: t.confidence, reverse=True)
    _identify_properties(targets, image_ary)

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

    try:
        offset_bboxes = detector_model.detect_all([box.image
                                                   for box in detector_crops])
    except IndexError:
        print('Error processing Darknet output...assuming no shapes detected.')
        offset_bboxes = []

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
        shape = Shape.NAS
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


def _identify_properties(targets, full_image, padding=15):

    for target in targets:

        x = int(target.x) - padding
        y = int(target.y) - padding
        w = int(target.width) + padding * 2
        h = int(target.height) + padding * 2
        blob_image = full_image[y:y + h, x:x + w]

        img = PIL.Image.fromarray(cv2.cvtColor(blob_image, cv2.COLOR_BGR2RGB))
        target.image = img

        try:
            target_color, alpha_color = _get_colors(blob_image)
            target.background_color = target_color
            target.alphanumeric_color = alpha_color
        except cv2.error:
            target.background_color = Color.NONE
            target.alphanumeric_color = Color.NONE


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

    # ColorCube((Hl, sl, vl), (Hu, Su, Vu))
    color_cubes = {
        "white": ColorCube((0, 0, 85), (359, 20, 100)),
        "black": ColorCube((0, 0, 0), (359, 100, 25)),
        "gray": ColorCube((0, 0, 25), (359, 5, 75)),
        "blue": ColorCube((180, 70, 70), (345, 100, 100)),
        "red": ColorCube((350, 70, 70), (359, 100, 65)),
        "green": ColorCube((100, 60, 30), (160, 100, 100)),
        "yellow": ColorCube((60, 50, 55), (75, 100, 100)),
        "purple": ColorCube((230, 40, 55), (280, 100, 100)),
        "brown": ColorCube((300, 38, 20), (359, 100, 40)),
        "orange": ColorCube((15, 70, 75), (45, 100, 100))
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
        cl_dists = [color_cubes[key].get_closest_distance((h, s, v))
                    for key in color_cubes]
        dist = [np.sqrt((p[0] * p[0]) + (p[1] * p[1]) + (p[2] * p[2]))
                for p in cl_dists]
        index = dist.index(min(dist)) + 1
        return Color(index)
    else:
        index = contains.index(True) + 1
        return Color(index)

    return Color.NONE

def load_images(imgs_path):
    images = []
    num = 0
    while(os.path.exists(imgs_path+"/ex"+str(num)+".png")):
        images.append(PIL.Image.open(imgs_path+"/ex"+str(num)+".png"))
        num += 1
    return images

def get_iou(bb1, bb2):

    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

if __name__ == '__main__':

    imgs_path = sys.argv[1]
    output = open("diagnostics.txt","w")
    imgs = load_images(imgs_path)
    pred_full = []
    times = []
    iou_thresh = .7;

    for image in imgs:
        start_time = datetime.now()
        pred_full.append(find_targets(image))
        times.append(datetime.now()-start_time)

    for i, pred_img in enumerate(pred_full):

        truths = open(imgs_path+"/ex"+str(i)+".txt", "r").readlines()

        pred_target_data = set()
        true_target_data = set()
        for line in truths:
            true_target_data.add(tuple(line.replace("\n","").split(" ")))

        for target in pred_img:
            shape = target.shape
            alphanumeric = target.alphanumeric
            x = target.x
            y = target.y
            w = target.width
            h = target.height
            pred_target_data.add((shape, alphanumeric, x, y, w, h))

        correct_preds = {}
        correct_ious = {}

        for truth in true_target_data:
        	for pred in pred_target_data:
        		true_x = float(truth[1])
        		true_w = float(truth[3])
        		true_y = float(truth[2])
        		true_h = float(truth[4])
        		true_alphanumeric = truth[0][-1]
        		true_shape = truth[0][0:truth[0].index("_")]
        		pred_x = pred[2]
        		pred_y = pred[3]
        		pred_w = pred[4]
        		pred_h = pred[5]
        		pred_alphanumeric = pred[1]
        		pred_shape =str(pred[0])[str(pred[0]).index(".")+1:].lower()
        		bb_true = {
        			'x1':true_x,
        			'x2':true_x+true_w,
        			'y1':true_y,
        			'y2':true_y+true_h
        		}
        		bb_pred = {
        			'x1':pred_x,
        			'x2':pred_x+pred_w,
        			'y1':pred_y,
        			'y2':pred_y+pred_h
        		}
        		iou = get_iou(bb_true, bb_pred)
        		#print("TRUTH - "+str((true_x, true_y, true_alphanumeric, true_shape)))
        		#print("PRED - "+str((pred_x, pred_y, pred_alphanumeric, pred_shape)))
        		if(iou > iou_thresh and 
        			pred_shape==true_shape and 
        			pred_alphanumeric==true_alphanumeric):
        			correct_preds[truth] = pred
        			correct_ious[truth] = str(pred) + " WITH IOU " + str(iou)

        false_negatives = true_target_data.difference(set(correct_preds.keys()))
        false_positives = pred_target_data.difference(set(correct_preds.values()))

        output.write("---- IMAGE ex"+str(i)+" ----\n")
        output.write("ACTUAL:\n")
        output.write(str(true_target_data)+"\n")
        output.write("PREDICTED:\n")
        output.write(str(pred_target_data)+'\n')
        output.write("TRUE POSITIVES:\n")
        output.write(str(set(correct_ious.values()))+"\n")
        output.write("FALSE NEGATIVES:\n")
        output.write(str(false_negatives)+"\n")
        output.write("FALSE POSITIVES:\n")
        output.write(str(false_positives)+"\n")
        output.write("TIME ELAPSED:\n")
        output.write(str(times[i])+"\n\n")

    output.close()