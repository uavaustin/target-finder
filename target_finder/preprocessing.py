"""Contains logic for finding and filtering blobs."""

import cv2
import numpy as np

from .types import Blob


def find_blobs(image, min_width=20, max_width=100, limit=100, padding=20):
    """Return the blobs found in an image.

    Note that by default, 100 blobs maximum will be returned, blobs
    are sorted with the largest area first.

    Args:
        image (PIL.Image): Image to find blobs in.
        min_width (int, optional): The minimum width of a blob in the
            x and y direction. Defaults to 20 pixels.
        max_width (int, optional): The maximum width of a blob in the
            x and y direction. Defaults to 100 pixels.
        limit (int, optional): The maximum number of blobs to return.
            Defaults to 100.
        padding (int, optional): Padding to use on each side when
            cropping. Note that padding will stop if the cropped
            image hits the image boundaries. Defaults to 20 pixels.

    Returns:
        List[Blob]: The list of blobs found.
    """

    # Find the edges in the image.
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    edges = cv2.Canny(cv_image, 200, 500)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, 1)

    # invert the image and find connected regions
    ret, labels = cv2.connectedComponents((255 - edges), connectivity=8)

    largest_label = -1  # the largest label (might not be > max_width)
    largest_size = -1
    large_labels = []  # all blobs to big to be a shape (> max_width)

    num_parts = np.max(labels) + 1
    for label in range(0, num_parts):

        label_img = (labels == label)
        size = np.count_nonzero(label_img)

        if size > largest_size:
            largest_label = label
            largest_size = size

        # extract bbox to find width and height
        rows = np.any(label_img, axis=1)
        cols = np.any(label_img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        min_dim = min(rmax - rmin, cmax - cmin)
        max_dim = max(rmax - rmin, cmax - cmin)

        if min_dim < min_width or max_dim > max_width:
            # hide this label
            large_labels.append(label)

    # create binary b/w image
    binary_thresh = np.full(edges.shape, 0, np.uint8)

    # highlight blob areas
    for label in range(0, num_parts):
        if label != large_labels and label not in large_labels:
            binary_thresh[labels == label] = 255

    # erode so a blob cnt better fits its image
    binary_thresh = cv2.erode(binary_thresh, kernel, 1)

    # generate contours
    _, contours, _ = cv2.findContours(binary_thresh, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)

    blobs = []

    # For each contour, find its position and size from a bounding
    # box, make sure it's not too small, and then go ahead and add to
    # the list of blobs.
    for cnt in contours:

        x, y, width, height = cv2.boundingRect(np.asarray(cnt))
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        if perimeter < 4 * area:
            has_mask = True
        else:
            has_mask = False

        # align the contour with the cropped image
        rel_cnt = np.array(cnt)
        rel_cnt[:, :, 0] -= max(x - padding, 0)
        rel_cnt[:, :, 1] -= max(y - padding, 0)

        # Add the blob to the list without the image.
        blob = Blob(x, y, width, height, None, has_mask, rel_cnt, edges)

        if _is_shape_like_blob(blob, min_width, max_width):
            blobs.append(blob)

    # Sort the contours with the largest area first.
    blobs.sort(key=lambda b: b.width * b.height, reverse=True)

    # Limit the number of blobs to return.
    blobs = blobs[:limit]

    # Add images now for all blobs remaining.
    for blob in blobs:
        blob.image = _crop_image(image, blob.x, blob.y, blob.width,
                                 blob.height, padding)

    return blobs


def _is_shape_like_blob(blob, min_width, max_width):
    """Verify that this could be a blob containing a shape"""
    width = blob.width
    height = blob.height

    # check provided restrictions
    if min(width, height) < min_width or max(width, height) > max_width:
        return False

    # TODO: Uncomment once "Close contours for masking images" is complete
    # must be closed cnt
    # if not blob.has_mask:
    #     return False

    # check bbox ratio
    size_ratio = max(width, height) / min(width, height)
    if size_ratio > 2:
        return False

    # check solidity
    hull = cv2.convexHull(blob.cnt)
    solidity = cv2.contourArea(blob.cnt) / cv2.contourArea(hull)
    if solidity < .5:
        return False

    return True


def _crop_image(image, x, y, width, height, padding):
    """Crop the image, but not past the boundaries."""
    return image.crop((
        max(x - padding, 0),
        max(y - padding, 0),
        min(x + width + padding, image.width),
        min(y + height + padding, image.height)
    ))
