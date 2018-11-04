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
with tf.gfile.GFile(graph_loc, 'rb') as f:
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

    # get rgb arrays for both colors then extract the alpha as b/w image
    primary_rgb, secondary_rgb = _get_color(blob)
    alpha_img = _extract_alpha(blob, secondary_rgb, primary_rgb)

    # convert rbg arrays to color names
    primary = _get_color_name(primary_rgb, None)
    secondary = _get_color_name(secondary_rgb, primary)

    if not hasattr(target_finder_model, '__version__'):
        # old code
        image_array = cropped_img.convert('RGB')
        predictions = tf_session.run(softmax_tensor,
                                     {'DecodeJpeg:0': image_array})
    else:
        # Manually do jpg preprocessing and send array as 'Placeholder:0'
        image_array = np.array(cropped_img.resize((299, 299)), np.float32)
        image_array /= 255.
        predictions = tf_session.run(softmax_tensor,
                                     {'Placeholder:0': [image_array]})

    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    shape = Shape[label_lines[top_k[0]]]
    confidence = float(predictions[0][top_k[0]])

    target = None

    if confidence >= min_confidence and shape != Shape.NAS:

        target = Target(blob.x, blob.y, blob.width, blob.height, shape=shape,
                        background_color=primary, alphanumeric_color=secondary,
                        image=blob.image, confidence=confidence)

    return target


def _get_color(blob):
    """Find the primary and seconday colors of the the blob"""
    (color_a, count_a), (color_b, count_b) = _find_main_colors(blob)

    # this assumes the shape will have more pixels than alphanum
    if count_a > count_b:
        primary, secondary = color_a, color_b
    else:
        primary, secondary = color_b, color_a

    return primary, secondary


def _extract_alpha(blob, color, not_color):
    """Extract the alphanumeric as a b/w image"""
    # get masked blob.image using contour
    mask_img = np.array(blob.image)
    mask = np.zeros(mask_img.shape[:2], dtype='uint8')
    cv2.drawContours(mask, [blob.cnt], -1, 255, -1)
    masked_image = cv2.bitwise_and(mask_img, mask_img, mask=mask)

    # find the diff between each pixel and each color
    diff = np.sum(np.square(masked_image[:, :, :] - color[:]), axis=2)
    diff_not = np.sum(np.square(masked_image[:, :, :] - not_color[:]), axis=2)

    # create base image
    isolated_img = np.zeros(mask_img.shape[:2], np.uint8)
    isolated_img[:, :] = 0

    # color pixels closer to color than not color
    alpha_x, alpha_y = np.nonzero(diff < diff_not * 4)  # * 4 b/c noise
    isolated_img[alpha_x, alpha_y] = 255

    # make rest of background black
    bg_x, bg_y, _ = np.where(masked_image[:, :] == [0, 0, 0])
    isolated_img[bg_x, bg_y] = 0

    # find connected components to further remove noise
    ret, labels = cv2.connectedComponents(isolated_img, 8)

    # extract the largest component, this should be the alpha
    largest_label = -1
    largest_cnt = -1

    num_parts = np.max(labels)
    for label in range(1, num_parts):
        cnt = np.count_nonzero(labels == label)
        if cnt > largest_cnt:
            largest_label = label
            largest_cnt = cnt

    isolated_img[labels != largest_label] = 0

    # convert to a 3d (color) image and flip black/white
    isolated_img_color = np.zeros(mask_img.shape, np.uint8)
    isolated_img_color[isolated_img == 0] = 255
    isolated_img_color[isolated_img != 0] = 0

    return PIL.Image.fromarray(isolated_img_color.astype('uint8'), 'RGB')


def _find_main_colors(blob):
    """Find the two main colors of the blob"""
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


def _get_color_name(requested_color, prev_color):
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
