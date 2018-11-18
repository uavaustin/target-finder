
from target_finder.classification import find_targets
from target_finder.types import Target, Shape, Color

import PIL.Image
import zipfile
import os

BENCHMARK_DIR = '.bench'

SHAPES_DIR = os.path.join(BENCHMARK_DIR, 'shapes')
SHAPES_ZIP = os.path.join(BENCHMARK_DIR, 'bench.zip')


def run_benchmarks():

    _ensure_bench_data()

    for shape_fn in os.listdir(SHAPES_DIR):

        shape, alpha, main_color, alpha_color, _ = shape_fn.split('-')
        image = PIL.Image.open(os.path.join(SHAPES_DIR, shape_fn))

        expected_target = _get_target(shape, alpha, main_color, alpha_color)

        targets = find_targets(image)

        if len(targets) > 0:

            print(targets[0])
            print(expected_target)
            print('\n')


def _get_target(shape_name, alpha, color_name, alpha_color_name):

    # temp
    color_name = color_name.replace('grey', 'gray')
    alpha_color_name = alpha_color_name.replace('grey', 'gray')

    shape = eval('Shape.' + shape_name.upper())
    main_color = eval('Color.' + color_name.upper())
    alpha_color = eval('Color.' + alpha_color_name.upper())

    return Target(x=0, y=0, width=10, height=10,
                  orientation=0.0, confidence=1,
                  shape=shape, background_color=main_color,
                  alphanumeric=alpha, alphanumeric_color=alpha_color)


def _ensure_bench_data():

    os.makedirs(BENCHMARK_DIR, exist_ok=True)

    # download

    if not os.path.isdir(SHAPES_DIR):

        zip_ref = zipfile.ZipFile(SHAPES_ZIP, 'r')
        zip_ref.extractall(SHAPES_DIR)
        zip_ref.close()
