
from target_finder.classification import find_targets
from target_finder.types import Target, Shape, Color

import urllib.request
import numpy as np
import random
import PIL.Image
import zipfile
import time
import os


SHAPES_DIR = os.path.join(os.path.dirname(__file__), 'data')
SHAPES_ZIP = os.path.join(os.path.dirname(__file__), 'bench.zip')
DATA_URL = 'https://bin.org/download/file'


def run_benchmarks(max_shapes=-1, shuffle=True):

    _ensure_bench_data()

    # Store results for each shape
    found_targets = []
    times = []
    confidences = []

    real_shapes = []
    actual_shapes = []

    real_colors = []
    actual_colors = []

    real_alphas = []
    actual_alphas = []

    shape_files = os.listdir(SHAPES_DIR)

    if shuffle:
        random.shuffle(shape_files)

    if max_shapes > 0:
        shape_files[:max_shapes]

    # classify every file
    for i, shape_fn in enumerate(shape_files):

        # calc % done
        complete = (i + 1) / len(shape_files) * 100
        print(f'\rRunning...{complete:.1f}% ({i}/{len(shape_files)})', end='')

        # load image
        shape, alpha, main_color, alpha_color, _ = shape_fn.split('-')
        image = PIL.Image.open(os.path.join(SHAPES_DIR, shape_fn))

        # get expected target object
        expected_target = _get_target(shape, alpha, main_color, alpha_color)

        start_time = time.time()
        targets = find_targets(image)
        end_time = time.time()

        if len(targets) > 0:

            # record real vs actual
            found_targets.append(1)
            target = targets[0]

            confidences.append(target.confidence)

            real_shapes.append(expected_target.shape.value)
            actual_shapes.append(target.shape.value)

            real_colors.append(expected_target.background_color.value)
            real_colors.append(expected_target.alphanumeric_color.value)
            actual_colors.append(target.background_color.value)
            actual_colors.append(target.alphanumeric_color.value)

            real_alphas.append(expected_target.alphanumeric)
            actual_alphas.append(target.alphanumeric)

        else:
            found_targets.append(0)

        times.append(end_time - start_time)

    print('...done.')
    print('=' * 30)

    # General Stats
    target_ident_score = np.mean(found_targets) * 100
    confidence = np.mean(confidences) * 100
    total_time = np.sum(times)
    avg_time = np.mean(times)

    print('\nGeneral -')

    print(f'   Targets Identified: {target_ident_score:.2f}%')
    print(f'   Avg. Confidence: {confidence:.2f}%')
    print(f'   Total Time: {total_time:.2f}s')
    print(f'   Mean Time: {avg_time:.2f}s')

    # Shape Stats
    _display_enum_stats('Shapes', real_shapes, actual_shapes, Shape)

    # Color Stats
    _display_enum_stats('Colors', real_colors, actual_colors, Color)

    # Alpha Stats
    print('\nAlphanumerics -')

    alpha_acc = np.mean(real_alphas == actual_alphas) * 100

    print(f'   Overall: {alpha_acc:.2f}%')


def _display_enum_stats(test_name, real, actual, type_enum):
    """Display the stats of an enum like color or shape"""

    # lists of ids
    real = np.array(real)
    actual = np.array(actual)

    overall_acc = np.mean(real == actual) * 100

    stats = {}

    for _, obj in type_enum.__members__.items():
        idxs = np.where(actual == obj.value)[0]
        if len(idxs) == 0:
            stats[obj.name] = -1
        else:
            # calculate precision
            prec = np.mean(real[idxs] == obj.value)
            stats[obj.name] = prec * 100

    print(f'\n{test_name} -')
    print(f'   Overall: {overall_acc:.2f}%')

    for name, prec in stats.items():
        if prec >= 0:
            print(f'     {name}: {prec:.2f}%')


def _get_target(shape_name, alpha, color_name, alpha_color_name):
    """Convert the params to a Target(...)"""

    shape = eval('Shape.' + shape_name.upper())
    main_color = eval('Color.' + color_name.upper())
    alpha_color = eval('Color.' + alpha_color_name.upper())

    return Target(x=0, y=0, width=10, height=10,
                  orientation=0.0, confidence=1,
                  shape=shape, background_color=main_color,
                  alphanumeric=alpha, alphanumeric_color=alpha_color)


def _ensure_bench_data():
    """Ensure that there is data to benchmark"""

    print('Locating data...', end='')

    if not os.path.isdir(SHAPES_DIR):

        print('fetching...', end='')

        urllib.request.urlretrieve(DATA_URL, SHAPES_ZIP)

        zip_ref = zipfile.ZipFile(SHAPES_ZIP, 'r')
        zip_ref.extractall(SHAPES_DIR)
        zip_ref.close()

    print('done.')


if __name__ == "__main__":
    run_benchmarks()
