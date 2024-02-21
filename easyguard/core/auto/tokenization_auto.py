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
""" Auto Tokenizer class."""

import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

from matplotlib import transforms
from transformers.tokenization_utils_base import TOKENIZER_CONFIG_FILE

from ...modelzoo.hub import AutoHubClass
from ...utils import (
    cache_file,
    file_exist,
    file_read,
    hf_name_or_path_check,
    lazy_model_import,
    logging,
    pretrained_model_archive_parse,
    sha256,
)
from . import BACKENDS, MODEL_ARCHIVE_CONFIG, MODEL_CONFIG_NAMES, MODELZOO_CONFIG, TOKENIZER_CONFIG_NAMES, VOCAB_NAME
from .auto_factory import _LazyAutoMapping
from .configuration_auto import CONFIG_MAPPING_NAMES, AutoConfig

logger = logging.get_logger(__name__)

if TYPE_CHECKING:
    # This significantly improves completion suggestion performance when
    # the transformers package is used with Microsoft's Pylance language server.
    TOKENIZER_MAPPING_NAMES: OrderedDict[str, Tuple[Optional[str], Optional[str]]] = OrderedDict()
else:
    TOKENIZER_MAPPING_NAMES = MODELZOO_CONFIG.get_mapping("tokenizer", "tokenizer_fast")

TOKENIZER_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, TOKENIZER_MAPPING_NAMES)

CONFIG_TO_TYPE = {v: k for k, v in CONFIG_MAPPING_NAMES.items()}


class AutoTokenizer:
    def __init__(self) -> None:
        raise EnvironmentError(
            "AutoTokenizer is designed to be instantiated "
            "using the `AutoTokenizer.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def from_config(cls, config_path: Union[str, Any], *inputs, **kwargs):
        # TODO (junwei.Dong): instantiate a tokenizer class from local path or config instance
        ...

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        region: Optional[str] = "CN",
        if_cache: Optional[bool] = False,
        *inputs,
        **kwargs,
    ):
        # TODO (junwei.Dong): refactor code just like AutoModel
        """

        Parameters
        ----------
        pretrained_model_name_or_path : str
            _description_
        """
        model_url = None
        server_name = None
        is_local = False
        config_path = None
        if pretrained_model_name_or_path not in MODEL_ARCHIVE_CONFIG:
            # if the `model_name_or_path` is not in `MODEL_ARCHIVE_CONFIG`, what we can do
            if os.path.exists(pretrained_model_name_or_path) and os.path.isdir(pretrained_model_name_or_path):
                model_config_path = file_exist(pretrained_model_name_or_path, MODEL_CONFIG_NAMES)
                assert (
                    model_config_path is not None
                ), f"please make sure the model config file exist in f{pretrained_model_name_or_path}"
                config_dict = file_read(model_config_path)
                model_type = config_dict.get("model_type", None)
                assert (
                    model_type is not None
                ), f"please check the config file in f{pretrained_model_name_or_path}, make sure the `model_type` key exists"
                is_local = True
                model_url = pretrained_model_name_or_path
            else:
                try:
                    from transformers import AutoTokenizer

                    return AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
                except:
                    raise KeyError(pretrained_model_name_or_path)
        else:
            model_archive = pretrained_model_archive_parse(
                pretrained_model_name_or_path,
                MODEL_ARCHIVE_CONFIG[pretrained_model_name_or_path],
                region,
            )
            model_type = model_archive.get("type", None)
            model_url = model_archive.get("url_or_path", None)
            server_name = model_archive.get("server", None)

        model_config = MODELZOO_CONFIG.get(model_type, None)
        assert (
            model_config is not None
        ), f"the target model `{model_type}` does not exist, please check the modelzoo or the config yaml~"

        backend = model_config.get("backend", None)
        assert backend in BACKENDS, f"backend should be one of f{BACKENDS}"
        backend_default_flag = False

        if backend == "hf":
            from .tokenization_auto_hf import HFTokenizer

            pretrained_model_name_or_path_ = (
                hf_name_or_path_check(
                    pretrained_model_name_or_path,
                    model_url,
                    model_type,
                )
                if not is_local
                else pretrained_model_name_or_path
            )

            return HFTokenizer.from_pretrained(pretrained_model_name_or_path_, *inputs, **kwargs)
        elif backend == "titan":
            # TODO (junwei.Dong): 支持特殊的titan模型
            raise NotImplementedError(backend)
        elif backend == "fex":
            # TODO (junwei.Dong): 支持特殊的fex模型
            raise NotImplementedError(backend)
        else:
            backend_default_flag = True

        if backend_default_flag:
            # just support base tokenizer
            # lazily obtain tokenizer class
            tokenizer_key = "tokenizer"
            tokenizer_name_tuple = MODELZOO_CONFIG[model_type][tokenizer_key]
            (
                tokenizer_module_package,
                tokenizer_module_name,
            ) = MODELZOO_CONFIG.to_module(tokenizer_name_tuple)
            tokenizer_class = lazy_model_import(tokenizer_module_package, tokenizer_module_name)
            extra_dict = {
                "server_name": server_name,
                "archive_name": pretrained_model_name_or_path,
                "model_type": model_type,
                "remote_url": model_url,
                "region": region,
            }
            AutoHubClass.kwargs = extra_dict
            # obtain vocab file path
            try:
                vocab_file_path = (
                    cache_file(
                        pretrained_model_name_or_path,
                        VOCAB_NAME,
                        if_cache=if_cache,
                        **extra_dict,
                    )
                    if not is_local
                    else file_exist(pretrained_model_name_or_path, VOCAB_NAME)
                )
            except:
                logger.info(
                    f"can not find vocab file, please generate tokenizer class based on tokenizer config file~"
                )
                vocab_file_path = None

            # obtain tokenizer config file path
            tokenizer_config_file_path = (
                cache_file(
                    pretrained_model_name_or_path,
                    TOKENIZER_CONFIG_NAMES,
                    if_cache=if_cache,
                    **extra_dict,
                )
                if not is_local
                else file_exist(pretrained_model_name_or_path, TOKENIZER_CONFIG_NAMES)
            )
            assert tokenizer_config_file_path is not None, f"tokenizer config file does not exist"

            tokenizer_config = file_read(tokenizer_config_file_path)
            tokenizer_config.update(kwargs)
            return tokenizer_class(vocab_file=vocab_file_path, **tokenizer_config, **extra_dict)
        ...
