# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
""" Auto Config class."""

import os
from collections import OrderedDict
from typing import List, Optional, Union

from genericpath import isfile

from ...modelzoo import MODELZOO_CONFIG
from ...modelzoo.configuration_utils import ConfigBase
from ...utils import CONFIG_NAME, cache_file, file_read, hf_name_or_path_check, lazy_model_import, logging
from . import MODEL_CONFIG_NAMES, MODELZOO_CONFIG

# TODO (junwei.Dong): 需要简化一下configuration_auto的逻辑

logger = logging.get_logger(__name__)

CONFIG_MAPPING_NAMES = MODELZOO_CONFIG.get_mapping("config")
MODEL_NAMES_MAPPING = MODELZOO_CONFIG.get_mapping("name")
SPECIAL_MODEL_TYPE_TO_MODULE_NAME = OrderedDict(
    [
        ("openai-gpt", "openai"),
        ("data2vec-audio", "data2vec"),
        ("data2vec-text", "data2vec"),
        ("data2vec-vision", "data2vec"),
        ("donut-swin", "donut"),
        ("maskformer-swin", "maskformer"),
        ("xclip", "x_clip"),
    ]
)


class AutoConfig:
    def __init__(self) -> None:
        raise EnvironmentError(
            "AutoConfig is designed to be instantiated "
            "using the `AutoConfig.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        config_name: Optional[str] = "config",
        if_cache: Optional[bool] = False,
        **kwargs,
    ):
        backend = kwargs.pop("backend", None)
        model_type = kwargs.pop("model_type", None)
        model_url = kwargs.pop("remote_url", None)
        backend_default_flag = False
        is_local = kwargs.pop("is_local", None)
        if backend == "default":
            backend_default_flag = True
        elif backend == "titan":
            # TODO (junwei.Dong): 支持特殊的titan模型
            raise NotImplementedError(backend)
        elif backend == "fex":
            # TODO (junwei.Dong): 支持特殊的fex模型
            raise NotImplementedError(backend)
        else:
            from .configuration_auto_hf import HFAutoConfig

            pretrained_model_name_or_path_ = (
                hf_name_or_path_check(
                    pretrained_model_name_or_path,
                    model_url,
                    model_type,
                )
                if not is_local
                else pretrained_model_name_or_path
            )

            return HFAutoConfig.from_pretrained(pretrained_model_name_or_path_, **kwargs)

        if backend_default_flag:
            model_config_name_tuple = MODELZOO_CONFIG[model_type][config_name]
            (
                model_config_module_package,
                model_config_module_name,
            ) = MODELZOO_CONFIG.to_module(model_config_name_tuple)
            # obtain config class
            model_config_class = lazy_model_import(model_config_module_package, model_config_module_name)
            extra_dict = {
                "model_type": model_type,
                "remote_url": model_url,
            }
            if os.path.isfile(pretrained_model_name_or_path):
                model_config_file_path = pretrained_model_name_or_path
            else:
                # obtain model config file path
                model_config_file_path = (
                    cache_file(
                        pretrained_model_name_or_path,
                        MODEL_CONFIG_NAMES,
                        if_cache=if_cache,
                        **extra_dict,
                    )
                    if not is_local
                    else kwargs["config_path"]
                )

            model_config = file_read(model_config_file_path)
            return model_config_class(**model_config)
