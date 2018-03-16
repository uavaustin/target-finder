"""Contains logic for finding and filtering blobs."""

from .types import Blob


def find_blobs(image, crop=True, min_width=20, limit=100):
    """Return the blobs found in an image.

    Note that by default, 100 blobs maximum will be returned, blobs
    are sorted with the largest area first.

    Args:
        image (PIL.Image): Image to find blobs in.
        crop (bool, optional): Whether or not to return a cropped
            image along with the blobs. Defaults to True.
        min_width (int, optional): The mimimum width of a blob in the
            x and y direction. Defaults to 20 pixels.
        limit (int, optional): The maximum number of blobs to return.

    Returns:
        List[Blob]: The list of blobs found.
    """

    # TODO: Implement.
    return []
