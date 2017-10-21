import os

import dotenv
import yaml


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
