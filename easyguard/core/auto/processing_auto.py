import os
from collections import OrderedDict
from typing import Any, Optional, Union

from ...modelzoo.hub import AutoHubClass
from ...utils import (
    file_exist,
    file_read,
    hf_name_or_path_check,
    lazy_model_import,
    logging,
    pretrained_model_archive_parse,
)
from ...utils.auxiliary_utils import cache_file
from . import BACKENDS, MODEL_ARCHIVE_CONFIG, MODEL_CONFIG_NAMES, MODELZOO_CONFIG, PROCESSOR_CONFIG_NAMES, VOCAB_NAME

logger = logging.get_logger(__name__)

PROCESSOR_MAPPING_NAMES = MODELZOO_CONFIG.get_mapping("processor")


class AutoProcessor:
    def __init__(self) -> None:
        raise EnvironmentError(
            "AutoTokenizer is designed to be instantiated "
            "using the `AutoTokenizer.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def from_config(cls, config_path: Union[str, Any], *inputs, **kwargs):
        # TODO (junwei.Dong): instantiate a processor class from local path or config instance
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
        """get processor for mml model

        Parameters
        ----------
        pretrained_model_name_or_path : str
            the name of pertrained models
        region : Optional[str], optional
            avaiable region, CN (China), VA (oversea), CN/VA (both), by default "CN"
        """

        """ >> initialize vars <<"""
        model_archive = None
        remote_url = None
        backend = None
        model_type = None
        model_config = None
        server_name = None
        region = region
        is_local = False
        processor_config_path = None
        vocab_path = None
        backend_default_flag = False

        extra_dict = OrderedDict()
        processor_dict = OrderedDict()

        """ >> get processor infomation <<"""
        if pretrained_model_name_or_path in MODEL_ARCHIVE_CONFIG:
            # parse model for integrating the url of targe server into model arhive config
            model_archive = pretrained_model_archive_parse(
                pretrained_model_name_or_path,
                MODEL_ARCHIVE_CONFIG[pretrained_model_name_or_path],
                region,
            )
            model_type = model_archive.get("type", None)
            remote_url = model_archive.get("url_or_path", None)
            server_name = model_archive.get("server", None)

            is_local = False
        elif os.path.isdir(pretrained_model_name_or_path):
            model_config_path = file_exist(pretrained_model_name_or_path, MODEL_CONFIG_NAMES)
            assert (
                model_config_path is not None
            ), f"please make sure the config file exist in f{pretrained_model_name_or_path}"

            config_dict_ = file_read(model_config_path)

            model_type = config_dict_.get("model_type", None)

            processor_config_path = file_exist(pretrained_model_name_or_path, PROCESSOR_CONFIG_NAMES)

            vocab_path = file_exist(pretrained_model_name_or_path, VOCAB_NAME)

            is_local = True
        else:
            logger.warning("can not found model location, load from huggingface...")

        """ >> preprocessing: download files << """
        if server_name:
            if not processor_config_path:
                processor_config_path = cache_file(
                    pretrained_model_name_or_path,
                    PROCESSOR_CONFIG_NAMES,
                    remote_url,
                    model_type,
                    if_cache=if_cache,
                    **kwargs,
                )
            if not vocab_path:
                vocab_path = cache_file(
                    pretrained_model_name_or_path,
                    VOCAB_NAME,
                    remote_url,
                    model_type,
                    if_cache=if_cache,
                    **kwargs,
                )

        """ >> load processor config class and model class <<"""

        processor = None

        # processor arguments that will be passed to processor class
        processor_dict.update({"vocab_path": vocab_path})
        # extra arguments will be set as processor extra argus
        extra_dict.update(
            {
                "server_name": server_name,
                "archive_name": pretrained_model_name_or_path,
                "model_type": model_type,
                "remote_url": remote_url,
                "region": region,
            }
        )

        if not model_type:
            logger.info("try to use transformers to load processor~")
            try:
                from transformers import AutoProcessor

                processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)
            except:  # noqa: E722
                raise KeyError(pretrained_model_name_or_path)
        else:
            model_config = MODELZOO_CONFIG.get(model_type, None)
            assert (
                model_config is not None
            ), f"the target model `{model_type}` does not exist, please check the modelzoo or the config yaml~"
            backend = model_config.get("backend", None)
            assert backend in BACKENDS, f"backend should be one of f{BACKENDS}"

            extra_dict["backend"] = backend

            if backend == "hf":
                from .processing_auto_hf import HFAutoProcessor

                pretrained_model_name_or_path_ = (
                    hf_name_or_path_check(pretrained_model_name_or_path, remote_url, model_type)
                    if not is_local
                    else pretrained_model_name_or_path
                )
                processor = HFAutoProcessor.from_pretrained(pretrained_model_name_or_path_, *inputs, **kwargs)
            elif backend == "titan":
                # TODO (junwei.Dong): 支持特殊的titan模型
                raise NotImplementedError(backend)
            elif backend == "fex":
                # TODO (junwei.Dong): 支持特殊的fex模型
                raise NotImplementedError(backend)
            else:
                backend_default_flag = True

            if backend_default_flag:
                processor_name_tuple = MODELZOO_CONFIG[model_type]["processor"]
                (
                    processor_module_package,
                    processor_module_name,
                ) = MODELZOO_CONFIG.to_module(processor_name_tuple)
                processor_class = lazy_model_import(processor_module_package, processor_module_name)
                AutoHubClass.kwargs = extra_dict

                processor_config = file_read(processor_config_path)
                processor_config.update(kwargs)
                processor = processor_class(**processor_config, **processor_dict, **extra_dict)
        """ >> processor post processing <<"""

        if processor:
            setattr(processor, "extra_args", extra_dict)

        return processor
