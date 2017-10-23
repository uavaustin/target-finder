import os

import dotenv
import yaml


TARGET_PROPERTIES = [
    'x',
    'y',
    'position_tol',
    'orientation',
    'orientation_tol',
    'shape',
    'background_color',
    'alphanumeric',
    'alphanumeric_color'
]


SHAPES = [
    'circle',
    'semicircle',
    'quarter_circle',
    'triangle',
    'square',
    'rectangle',
    'trapezoid',
    'pentagon',
    'hexagon',
    'heptagon',
    'octagon',
    'star',
    'cross'
]


COLORS = [
    'white',
    'black',
    'gray',
    'red',
    'blue',
    'green',
    'yellow',
    'purple',
    'brown',
    'orange'
]


def collect_images():
    """Collect images from the e2e testing directory.

    The directory must be specified in the TF_TESTING_IMG environment
    variable.

    Raises:
        Exception: If the no directory is specified

    Ruturns:
        List<>: List of
    """

    config = _read_target_config()
    config = _listify_targets(config)

    _check_target_properties(config)

    return config


def _read_target_config():
    """Get the filenames from targets.yml"""
    # Loading settings from .env if they're there
    dotenv_file = dotenv.find_dotenv()

    if dotenv_file:
        dotenv.load_dotenv(dotenv_file)

    testing_dir = os.environ.get('TF_TESTING_IMG')

    if not testing_dir:
        raise Exception('Testing directory not specified. Cannot run tests.')

    testing_dir = os.path.normpath(os.path.expanduser(testing_dir))
    target_config = os.path.join(testing_dir, 'targets.yml')

    if not os.path.exists(testing_dir):
        raise Exception(testing_dir + ' does not exist')

    if not os.path.isdir(testing_dir):
        raise Exception(testing_dir + ' is not a directory')

    if not os.path.exists(target_config):
        raise Exception(target_config + ' does not exist')

    if not os.path.isfile(target_config):
        raise Exception(target_config + ' is not a file')

    with open(target_config, 'r') as f:
        return yaml.load(f)


def _listify_targets(config):
    "Make sure targets are in lists"
    for filename in config.keys():
        # If no targets were listed, it should be an empty list
        if config[filename] == None:
            config[filename] = []
        # Otherwise, if they aren't lists make them lists
        elif not isinstance(config[filename], list):
            config[filename] = [config[filename]]

    return config


def _check_target_properties(config):
    "Throw an error if an invalid target was listed"
    for filename in config.keys():
        for target in config[filename]:
            if not all(prop in TARGET_PROPERTIES for prop in target.keys()):
                raise Exception('Invalid target property for ' + filename)

            if 'shape' in target.keys() and not target['shape'] in SHAPES:
                raise Exception('Invalid shape for ' + filename)

            if 'background_color' in target.keys() and \
                    not target['background_color'] in COLORS:
                print(target['background_color'])
                print(COLORS)

                raise Exception('Invalid background_color for ' + filename)

            if 'alphanumeric_color' in target.keys() and \
                    not target['alphanumeric_color'] in COLORS:
                raise Exception('Invalid alphanumeric_color for ' + filename)
