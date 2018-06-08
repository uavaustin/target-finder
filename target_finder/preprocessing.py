"""Contains logic for finding and filtering blobs."""

import cv2
import numpy as np

from .types import Blob


def find_blobs(image, mask_img, min_width=20, max_length=100, limit=100, padding=10):
    """Return the blobs found in an image.

    Note that by default, 100 blobs maximum will be returned, blobs
    are sorted with the largest area first.

    Args:
        image (PIL.Image): Image to find blobs in.
        padding (int, optional): Padding to use on each side when
            cropping. Note that padding will stop if the cropped
            image hits the image boundaries. Defaults to 10 pixels.
        min_width (int, optional): The minimum width of a blob in the
            x and y direction. Defaults to 20 pixels.
        limit (int, optional): The maximum number of blobs to return.

    Returns:
        List[Blob]: The list of blobs found.
    """

    # First, we find the edges in the image.
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    edges = cv2.Canny(cv_image, 200, 500)
    #kernel = np.ones((3,3), np.uint8)
    #edges = cv2.dilate(edges, kernel, 1)
    #edges = cv2.erode(edges, kernel, 1)

    # Next, we find the contours according to the threshold.
    ret, thresh = cv2.threshold(edges, 127, 255, 0)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_NONE)

    blobs = []

    # For each contour, we find its position and size from a bounding
    # box, we make sure it's not too small, and then we'll go ahead
    # and add to the list of blobs.
    for cnt in contours:

        Mask = False

        x, y, width, height = cv2.boundingRect(np.asarray(cnt))
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        if perimeter < 3*area:
            Mask = True

        if (width < min_width or height < min_width or height > max_length or width > max_length): continue

        # Adding the blob to the list. We're not adding images yet
        # since we don't want to do that if we're not going to keep
        # the blob.
        blobs.append(Blob(x, y, width, height, None, mask_img, Mask, cnt))

    # Sorting the contours with the ones with the largest area first.
    blobs.sort(key=lambda b: b.width * b.height, reverse=True)

    # Limiting the number of blobs we're returning.
    blobs = blobs[:limit]

    # Adding images now for all blobs remaining.
    for blob in blobs:
        blob.image = _crop_image(image, blob.x, blob.y, blob.width,
                                  blob.height, padding)

    return blobs


def _crop_image(image, x, y, width, height, padding):
    """Crops the image, but doesn't go past the boundaries."""
    return image.crop((
        max(x - padding, 0),
        max(y - padding, 0),
        min(x + width + padding, image.width),
        min(y + height + padding, image.height)
    ))
