"""Contains logic for finding targets in blobs."""

from .preprocessing import find_blobs
from .types import Color, Shape, Target


# TODO: Make a Tensorflow session we can use. There should be two
#       seperate lookups for models. The first one will see if we
#       have a user-made one, if so, we'll use that one. Otherwise,
#       we'll use a default model that ships with the library.


def find_targets(image=None, blobs=None, min_confidence=0.95, limit=10):
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


def _do_classify(blob, min_confidence):
    """Perform the classification on a blob.

    Returns None if it's not a target.
    """

    # TODO: Implement.
    return None
