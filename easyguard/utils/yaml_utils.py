import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml
from easydict import EasyDict

from .hdfs_utils import hdfs_open
from .logging import get_logger
from .re_exp import E_STR
from .type_utils import typecheck

logger = get_logger(__name__)

"""Operators for yaml"""


E_RE = re.compile(E_STR)


def yaml_check(data: Dict[str, Any]):
    """check the str such as `1e-6`

    Parameters
    ----------
    data : Dict[str, Any]
        the converted json data
    """
    for key_, value_ in data.items():
        if isinstance(value_, dict):
            yaml_check(value_)
        else:
            if isinstance(value_, str) and E_RE.match(value_) is not None:
                data[key_] = float(value_)


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
    if not path.startswith("hdfs://"):
        with open(path, "r") as yaml_file:
            data = yaml.full_load(yaml_file)
    else:
        with hdfs_open(path, "rb") as hdfs_reader:
            logger.info(f"start to read `{path}` file from hdfs")
            data = yaml.full_load(hdfs_reader)
            logger.info(f"read `{path}` successfully")

    yaml_check(data)

    return EasyDict(data)


def load_json(path: str) -> Dict[str, Any]:
    """load json file, convert json to dict

    Parameters
    ----------
    path : str
        the json file path

    Returns
    -------
    Dict[str, Any]
        dict data
    """
    if not path.startswith("hdfs://"):
        with open(path, "r") as json_file:
            data = json.load(json_file)
    else:
        with hdfs_open(path, "rb") as hdfs_reader:
            logger.info(f"start to read `{path}` file from hdfs")
            data = json.load(hdfs_reader)
            logger.info(f"read `{path}` successfully")

    return EasyDict(data)


@typecheck(str)
def file_read(path: str) -> Dict[str, Any]:
    """支持json和yaml的读写

    Parameters
    ----------
    path : str
        target file path

    Returns
    -------
    Dict[str, Any]
        dict data
    """
    if path.endswith(".json"):
        return load_json(path)
    elif path.endswith(".yaml"):
        return load_yaml(path)
    return None


def json2yaml(path: str, data: Dict[str, Any]):
    """output json data to yaml file

    Parameters
    ----------
    path : str
        the target output yaml file path
    data : Dict[str, Any]
        the json data
    """
    with open(path, "w", encoding="utf-8") as writer:
        yaml.dump(data, writer)


@dataclass
class YamlConfig(ABC):
    path: str
    config: Dict[str, Any]
    leafs: List[tuple]
    name: str = field(default="Models Config", repr=False)
    docstring: str = field(default="All models config settings", repr=False)

    @abstractmethod
    def check(self):
        ...

    @classmethod
    def yaml_reader(
        cls,
        path: str,
        name: str = "modelzoo config",
        docstring: str = "it is about the modelzoo setting",
    ):
        """Instantiate a `YamlConfig` object from a yaml config

        Parameters
        ----------
        path : str
            a yaml config file
        name : str, optional
            config name, by default "modelzoo config"
        docstring : str, optional
            document string, by default "it is about the modelzoo setting"

        Returns
        -------
        YamlConfig
            return a YamlConfig object
        """
        config_data = load_yaml(path)

        dfs_leafs = []
        cls.dfs_decouple(config_data, dfs_leafs)
        yamlconfig_ = cls(path, config_data, dfs_leafs, name=name, docstring=docstring)
        return yamlconfig_

    @classmethod
    def dfs_decouple(
        cls,
        data: Dict[str, Any],
        save_data: List[tuple],
        prefix: Optional[str] = None,
    ):
        """find all the leafs about a json data

        Parameters
        ----------
        data : Dict[str, Any]
            a json data
        save_data : List[tuple]
            _description_
        prefix : Optional[str], optional
            used to save the path of each json key, by default None
        """
        for key_item, value in data.items():
            prefix_ = f"{prefix}.{key_item}" if prefix else f"{key_item}"
            if isinstance(value, dict):
                cls.dfs_decouple(value, save_data, prefix_)
            else:
                save_data.append((prefix_, key_item, value))

    def __getitem__(self, key: str) -> Any:
        """used to read the target key value

        Example:
            {'name': 'Jace', 'body':{'weight': 130}}
            to get the weight, we can use the key `body.weight`
        Parameters
        ----------
        key : str
            a key, for example, 'models.bert.config'

        Returns
        -------
        Any
            the target value
        """
        keys_ = key.split(".")
        data_ = self.config
        for index, key_item in enumerate(keys_):
            data_ = data_.get(key_item, None)
            assert (
                data_ is not None
            ), f'the target model `{".".join(keys_[:index])}` does not exist, please check the modelzoo or the config yaml~'  # noqa: E501

        return data_

    # def __repr__(self) -> str:
    #     return f'{self.docstring}'

    def __str__(self) -> str:
        return f"{self.docstring}"


if __name__ == "__main__":
    ...
