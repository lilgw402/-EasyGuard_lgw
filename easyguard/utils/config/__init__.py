import os
from typing import Any, Dict

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    """load yaml file, convert yaml to dict

    Parameters
    ----------
    path : str
        the yaml file path

    Returns
    -------
    Dict[str, Any]
        dict data
    """

    with open(path, "r") as yaml_file:
        data = yaml.full_load(yaml_file)
    return data


LOGGER_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "logger.yaml")

LOGGER_CONFIG = load_yaml(LOGGER_CONFIG_PATH)
