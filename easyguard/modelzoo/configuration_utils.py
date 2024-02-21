# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Configuration base class and utilities."""
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers.configuration_utils import *

from .. import __version__
from ..utils import logging, typecheck

logger = logging.get_logger(__name__)


# TODO (junwei.Dong): 一些通用的关于config的操作可以注册进该基类中去, 自己开发的config需要继承该基类
# EasyGuard config base class
@dataclass
class ConfigBase(ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    @typecheck(object, str)
    def __getitem__(self, key: str) -> Any:
        """like a dict class"""
        return self.__dict__[key]

    @typecheck(object, str)
    def get(self, key: str, default):
        return self.__dict__.get(key, default)

    def asdict(self) -> Dict[str, Any]:
        return asdict(self)

    @typecheck(object, dict)
    def update(self, data: Dict[str, Any]):
        """update arguments
        example:

        >>> self.model_name
        output:
            Bert
        >>> data = {'model_name': 'bert', 'type': 'NLP'}
        >>> self.update(data)
        >>> self.model_name
        output:
            bert
        >>> self.type
        output:
            NLP
        output
        Parameters
        ----------
        data : Dict[str, Any]
            _description_
        """
        self.__dict__.update(data)

    def config_update_for_pretrained(self, **kwargs):
        """pretrained config modify"""
