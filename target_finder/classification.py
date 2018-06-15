"""Contains logic for finding targets in blobs."""

from .preprocessing import find_blobs
from .types import Color, Shape, Target
import cv2
from PIL import Image
import numpy as np
#import pytesseract
import os
import scipy.cluster, scipy.misc
import tensorflow as tf
import webcolors
from pkg_resources import resource_filename

graph_loc = resource_filename(__name__, 'data/retrained_graph.pb')
labels_loc = resource_filename(__name__, 'data/retrained_labels.txt')
#pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#warnings.filterwarnings("ignore")

#tf.device('/cpu:0')

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line in tf.gfile.GFile(labels_loc)]

# Unpersists graph from file
with tf.gfile.FastGFile(graph_loc, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

#config = tf.ConfigProto(
#    device_count = {'GPU': 0}
#)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

def find_targets(image=None, blobs=None, min_confidence=0.85, limit=10):
    """Returns the targets found in an image.

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
            use (0 <= confidence <= 1). Default is 0.95.
        limit (int, optional): The maximum number of targets to
            return.

    Returns:
        List[Target]: The list of targets found.
    """

    # Check that if we don't have an image passed, that the each
    # blobs have their own image.
    if image is None and blobs is None:
        raise Exception('Blobs must be provided if an image is not.')

    # If we didn't get blobs, then we'll find them.
    if blobs is None:
        blobs = find_blobs(image)

    targets = []

    # Try and find a target for each blob, if it exists then register
    # it. Stop if we hit the limit.
    for blob in blobs:
        if len(targets) == limit: break

        target = _do_classify(blob, min_confidence)

        if target is not None:
            # TODO - prevent duplicates in case a target is too
            #        similar to another one.
            targets.append(target)

    # Sorting with highest confidence first.
    targets.sort(key=lambda t: t.confidence, reverse=True)

    return targets


def get_alpha(blob, Target):

    angle = 0
    delta_angle = 10

    mask_img = np.array(blob.image)

    alpha_mask = np.zeros(mask_img.shape[:2], dtype='uint8')
    cv2.drawContours(alpha_mask, blob.cnt, -1, 255, -1)
    alphaedges = cv2.bitwise_and(blob.edges, blob.edges, mask=alpha_mask)
    cv2.drawContours(alphaedges, blob.cnt, -1, (0,0,0), 2)
    #width, height = dst.shape[:2]

    while angle < 360:
        img = alphaedges
        image_array = img.rotate(angle)
        #data = pytesseract.image_to_data(image_array)
        text = image_to_string(image_array)
        angle += delta_angle

        if text in alphas:
            Target.alphanumeric = text
            Target.orientation = angle
            break



def get_color_name(requested_color, prev_color, colors_set):

    color_codes = {}
    i = 0

    # Makes sure alpha color and shape color are different
    if prev_color is not None:
        for key, name in colors_set.items():
            if name == prev_color:
                color_codes[i] = key
                i = i+1
        for i in color_codes:
            del colors_set[color_codes[i]]
    min_colors = {}

    # Finds closest color with a given RGB value
    for key, name in colors_set.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name

    return min_colors[min(min_colors.keys())]

def get_color(blob):
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

    mask_img = np.array(blob.image)

    dst = mask_img
    width, height = dst.shape[:2]

    if width > height:
        y1 = blob.y
        y2 = blob.y + blob.height
        x1 = blob.x
        x2 = blob.x + blob.width
    else:
        x1 = blob.y
        x2 = blob.y + blob.height
        y1 = blob.x
        y2 = blob.x + blob.width

    if blob.has_mask:
        mask = np.zeros(mask_img.shape[:2], dtype='uint8')
        cv2.drawContours(mask, [blob.cnt], -1, 255, -1)
        dst = cv2.bitwise_and(mask_img, mask_img, mask=mask)
    else:
        y1 = y1 + 5
        y2 = y2 - 5
        x1 = x1 + 5
        x2 = x2 - 5


    cropped_img = Image.fromarray(dst)
    #cropped_img = dst

    cropped_img.crop((x1, y1, x2, y2))

    ar = scipy.misc.fromimage(cropped_img)
    dim = ar.shape
    ar = ar.reshape(scipy.product(dim[:2]), dim[2])
    codes, dist = scipy.cluster.vq.kmeans(ar.astype(float), 3)
    primary = get_color_name(codes[0].astype(int), None, colors_set)
    if len(codes) > 1:
        secondary = get_color_name(codes[1].astype(int), primary, colors_set)
    else:
        secondary = Color.NONE
    # Ignores black mask for color detection, returns most prominent color as shape
    if primary == None:
        primary = secondary
        secondary = Color.NONE
    if secondary == Color.NONE and len(codes) > 2:
        tertiary = get_color_name(codes[2].astype(int), secondary, colors_set)
        secondary = tertiary
    return primary, secondary




def _do_classify(blob, min_confidence):
    """Perform the classification on a blob.

    Returns None if it's not a target.
    """

    cropped_img = blob.image

    image_array = cropped_img.convert('RGB')
    predictions = sess.run(softmax_tensor, {'DecodeJpeg:0': image_array})

    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    shape = Shape[label_lines[top_k[0]]]
    confidence = predictions[0][top_k[0]]

    if confidence < min_confidence or shape == Shape.NAS:
        return None
    else:
        primary, secondary = get_color(blob)
        shape = Target(blob.x, blob.y, blob.width, blob.height, shape=shape, background_color=primary, alphanumeric_color=secondary, image=blob.image, confidence=confidence)
        #get_alpha(blob, shape)
        return shape
