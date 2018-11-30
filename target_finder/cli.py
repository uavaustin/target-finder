"""Contains functions for cli subcommands."""

import argparse
import json
import os
import sys

import PIL.Image
import tensorflow as tf
import target_finder_model

from .classification import find_targets
from .preprocessing import find_blobs
from .version import __version__


# Create the top level parser.
parser = argparse.ArgumentParser()
parser.add_argument('-v', '--version', action='store_true',
                    help='show the version and exit')
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

# Parser for the targets subcommand.
target_parser = subparsers.add_parser('targets', help='finds the targets in '
                                                      'images')
target_parser.add_argument('filename', type=str, nargs='+',
                           help='the images or image directories')
target_parser.add_argument('-o', '--output', type=str, action='store',
                           default='.', help='output directory (defaults to '
                                             'current dir)')
target_parser.add_argument('--min-confidence', type=float, action='store',
                           default=0.85, help='confidence level for '
                                              'classification (default: 0.85)')
target_parser.add_argument('--limit', type=int, dest='limit', action='store',
                           default=10, help='maximum number of blobs to find '
                                            'per image (default: 10)')


def run(args=None):
    """Dispatch the correct subcommand."""
    args = parser.parse_args(args)

    # Print the version of this library and the model if requested.
    if args.version:
        print_version()
        return

    args.func(args)


def print_version():
    if hasattr(target_finder_model, '__version__'):
        model_version = target_finder_model.__version__
    else:
        model_version = '0.1.0'

    print(f'target-finder v{__version__} with target-finder-model '
          f'v{model_version} (tensorflow v{tf.__version__})')


def run_blobs(args):
    """Run the blobs subcommand."""
    blob_num = 0

    # Create the output directory if it doesn't already exist.
    os.makedirs(args.output, exist_ok=True)

    for filename in _list_images(args.filename):
        image = PIL.Image.open(filename)

        blobs = find_blobs(image, min_width=args.min_width,
                           max_width=args.max_width, limit=args.limit,
                           padding=args.padding)

        # Save each blob found with an incrementing number.
        for blob in blobs:
            print('Saving blob #{:06d} from {:s}'.format(blob_num, filename))

            basename = 'blob-{:06d}.jpg'.format(blob_num)
            blob.image.save(os.path.join(args.output, basename))

            blob_num += 1


def run_targets(args):
    """Run the targets subcommand."""
    target_num = 0

    # Create the output directory if it doesn't already exist.
    os.makedirs(args.output, exist_ok=True)

    for filename in _list_images(args.filename):
        image = PIL.Image.open(filename)

        targets = find_targets(image, min_confidence=args.min_confidence,
                               limit=args.limit)

        # Save each target found with an incrementing number.
        for target in targets:
            print('Saving target #{:06d} from {:s}'.format(target_num,
                                                           filename))

            basename_image = 'target-{:06d}.jpg'.format(target_num)
            basename_meta = 'target-{:06d}.json'.format(target_num)

            filename_image = os.path.join(args.output, basename_image)
            filename_meta = os.path.join(args.output, basename_meta)

            target.image.save(filename_image)
            _save_target_meta(filename_meta, filename, target)

            target_num += 1


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


def _save_target_meta(filename_meta, filename_image, target):
    """Save target metadata to a file."""
    with open(filename_meta, 'w') as f:
        meta = {
            'x': target.x,
            'y': target.y,
            'width': target.width,
            'height': target.height,
            'orientation': target.orientation,
            'shape': target.shape.name.lower(),
            'background_color': target.background_color.name.lower(),
            'alphanumeric': target.alphanumeric,
            'alphanumeric_color': target.alphanumeric_color.name.lower(),
            'image': filename_image,
            'confidence': target.confidence
        }

        json.dump(meta, f, indent=2)


# Set the functions to run for each subcommand. If a subcommand was
# not provided, print the usage message and set the exit code to 1.
blob_parser.set_defaults(func=run_blobs)
target_parser.set_defaults(func=run_targets)
parser.set_defaults(func=lambda _: parser.print_usage() or sys.exit(1))
