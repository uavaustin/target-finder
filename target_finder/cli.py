"""Contains functions for cli subcommands."""

import argparse
import os
import sys

import PIL.Image

from .preprocessing import find_blobs


# Create the top level parser.
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest='subcommand', title='subcommands')

# Parser for the blobs subcommand.
blob_parser = subparsers.add_parser('blobs', help='finds the interesting '
                                                  'blobs in images')
blob_parser.add_argument('filename', type=str, nargs='+',
                         help='the images or image directories')
blob_parser.add_argument('-o', '--output', type=str, action='store',
                         default='.', help='output directory (defaults to '
                                           'current dir)')
blob_parser.add_argument('--min-width', type=int, action='store', default=20,
                         help='minimum width a blob must be in the horizontal '
                              'and vertical directions (default: 20)')
blob_parser.add_argument('--max-width', type=int, action='store', default=100,
                         help='maximum width a blob can be in the horizontal '
                              'and vertical directions (default: 100)')
blob_parser.add_argument('--limit', type=int, dest='limit', action='store',
                         default=100, help='maximum number of blobs to find '
                                           'per image (default: 100)')
blob_parser.add_argument('--padding', type=int, action='store', default=20,
                         help='how much space to leave around blobs on each '
                              'side (default: 20 pixels)')


def run():
    """Dispatch the correct subcommand."""
    args = parser.parse_args()
    args.func(args)


def run_blobs(args):
    """Run the blobs subcommand."""
    blob_num = 0

    # Create the output directory if it doesn't already exist.
    os.makedirs(args.output, exist_ok=True)

    for filename in _list_images(args.filename):
        image = PIL.Image.open(filename)
        mask_img = cv2.imread(filename)

        blobs = find_blobs(image, mask_img, min_width=args.min_width,
                           max_width=args.max_width, limit=args.limit,
                           padding=args.padding)

        # Save each blob found with an incrementing number.
        for blob in blobs:
            print('Saving blob #{:06d} from {:s}'.format(blob_num, filename))

            basename = 'blob-{:06d}.jpg'.format(blob_num)
            blob.image.save(os.path.join(args.output, basename))

            blob_num += 1


def _list_images(filenames):
    """Turn the list of filenames into a list of images."""
    images = []

    for filename in filenames:
        # If this is a normal filename, add it to the list directly.
        if os.path.isfile(filename):
            images.append(filename)

        # If this is a directory, add the files ending with .jpg or
        # .jpeg (case-insensitive) to the list.
        elif os.path.isdir(filename):
            for inner_filename in os.listdir(filename):
                if inner_filename.lower().endswith('.jpg') or \
                        inner_filename.lower().endswith('.jpeg'):
                    images.append(os.path.join(filename, inner_filename))

        # If it's not either above, exit.
        else:
            print('Bad filename: "{:s}".'.format(filename))
            sys.exit(1)

    # There's a problem if we can't find any images.
    if images == []:
        print('No images found.')
        sys.exit(1)

    return images


# Set the functions to run for each subcommand. If a subcommand was
# not provided, print the usage message and set the exit code to 1.
blob_parser.set_defaults(func=run_blobs)
parser.set_defaults(func=lambda _: parser.print_usage() or sys.exit(1))
