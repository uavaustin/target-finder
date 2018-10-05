"""Contains logic for finding targets in blobs."""

import os
from pkg_resources import resource_filename

import cv2
import numpy as np
import os
import PIL.Image
import sklearn.cluster
import scipy.misc
import target_finder_model
import tensorflow as tf
import webcolors

from .preprocessing import find_blobs
from .types import Color, Shape, Target


graph_loc = target_finder_model.graph_file
labels_loc = target_finder_model.labels_file

# Make Tensorflow quieter.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load label file and strip off newlines.
label_lines = [line.rstrip() for line in tf.gfile.GFile(labels_loc)]

# Register the graph with tensorflow.
with tf.gfile.FastGFile(graph_loc, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

    tf.import_graph_def(graph_def, name='')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

tf_session = tf.Session(config=config)
softmax_tensor = tf_session.graph.get_tensor_by_name('final_result:0')


def find_targets(image=None, blobs=None, min_confidence=0.85, limit=10):
    """Return the targets found in an image.

    Targets are returned in the order of highest confidence. Once the
    limit is hit, classification will stop and just take the first
    ones.

    Either the image or blobs should be provided. If just passing in
    an image, all the defaults will be used when finding blobs.

    Args:
        image (PIL.Image, optional): The image to use, this must be
            provided if no blobs are passed in.
        blobs (List[Blob], optional): The list of blobs to use if
            they've already been found. If None is passed, then
            find_blobs() will be called prior to classification.
            Default is None.
        min_confidence (float, optional): Confidence threshold to
            use (0 <= confidence <= 1). Default is 0.85.
        limit (int, optional): The maximum number of targets to
            return.

    Returns:
        List[Target]: The list of targets found.
    """

    # Check that when is not an image passed, the each blob have
    # their own image.
    if image is None and blobs is None:
        raise Exception('Blobs must be provided if an image is not.')

    # If there is not a tensorflow session because of a missing graph
    # or labels, then there's nothing to do.
    if tf_session is None:
        return []

    # If we didn't get blobs, then we'll find them.
    if blobs is None:
        blobs = find_blobs(image)

    targets = []

    # Try and find a target for each blob, if it exists then register
    # it. Stop if we hit the limit.
    for blob in blobs:
        if len(targets) == limit:
            break

        target = _do_classify(blob, min_confidence)

        if target is not None:
            # TODO - prevent duplicates in case a target is too
            #        similar to another one.
            targets.append(target)

    # Sorting with highest confidence first.
    targets.sort(key=lambda t: t.confidence, reverse=True)

    return targets


def _do_classify(blob, min_confidence):
    """Perform the classification on a blob.

    Returns None if it's not a target.
    """

    cropped_img = blob.image

    image_array = cropped_img.convert('RGB')
    predictions = tf_session.run(softmax_tensor, {'DecodeJpeg:0': image_array})

    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    shape = Shape[label_lines[top_k[0]]]
    confidence = float(predictions[0][top_k[0]])

    target = None

    if confidence >= min_confidence and shape != Shape.NAS:
        primary, secondary = _get_color(blob)
        target = Target(blob.x, blob.y, blob.width, blob.height, shape=shape,
                        background_color=primary, alphanumeric_color=secondary,
                        image=blob.image, confidence=confidence)

    return target


def _get_color(blob):

    colors_set = {
        '#000000': None,
        '#000001': Color.BLACK,
        '#ffffff': Color.WHITE,
        '#407340': Color.GREEN,
        '#94ff94': Color.GREEN,
        '#00ff00': Color.GREEN,
        '#008004': Color.GREEN,
        '#525294': Color.BLUE,
        '#7f7fff': Color.BLUE,
        '#0000ff': Color.BLUE,
        '#000087': Color.BLUE,
        '#808080': Color.GRAY,
        '#994c00': Color.BROWN,
        '#e1dd68': Color.YELLOW,
        '#fffc7a': Color.YELLOW,
        '#fff700': Color.YELLOW,
        '#d2cb00': Color.YELLOW,
        '#d8ac53': Color.ORANGE,
        '#FFCC65': Color.ORANGE,
        '#ffa500': Color.ORANGE,
        '#d28c00': Color.ORANGE,
        '#bc3c3c': Color.RED,
        '#ff5050': Color.RED,
        '#ff0000': Color.RED,
        '#9a0000': Color.RED,
        '#800080': Color.PURPLE
    }

    (color_a, count_a), (color_b, count_b) = _find_main_colors(blob)

    # this assumes the shape will have more pixels than alphanum
    if count_a > count_b:
        primary = _get_color_name(color_a, None, colors_set)
        secondary = _get_color_name(color_b, primary, colors_set)
    else:
        primary = _get_color_name(color_b, None, colors_set)
        secondary = _get_color_name(color_a, primary, colors_set)

    return primary, secondary


def _find_main_colors(blob):

    mask_img = np.array(blob.image)  # the image w/the mask applied

    mask = np.zeros(mask_img.shape[:2], dtype='uint8')  # the mask itself

    # create mask
    cv2.drawContours(mask, [blob.cnt], -1, 255, -1)

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


def _get_color_name(requested_color, prev_color, colors_set):
    color_codes = {}
    i = 0

    # Makes sure alpha color and shape color are different.
    if prev_color is not None:
        for key, name in colors_set.items():
            if name == prev_color:
                color_codes[i] = key
                i = i + 1
        for i in color_codes:
            del colors_set[color_codes[i]]

    min_colors = {}

    # Find closest color with a given RGB value.
    for key, name in colors_set.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name

    return min_colors[min(min_colors.keys())]
