import os
from collections import OrderedDict
from typing import Any, Optional, Union

from ...modelzoo.hub import AutoHubClass
from ...utils import (
    cache_file,
    file_exist,
    file_read,
    hf_name_or_path_check,
    lazy_model_import,
    logging,
    pretrained_model_archive_parse,
)
from . import BACKENDS, IMAGE_PROCESSOR_CONFIG_NAMES, MODEL_ARCHIVE_CONFIG, MODEL_CONFIG_NAMES, MODELZOO_CONFIG

IMAGE_PROCESSOR_MAPPING_NAMES = MODELZOO_CONFIG.get_mapping("image_processor")

logger = logging.get_logger(__name__)


class AutoImageProcessor:
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
        """instantiate a processor class based on `pretrained_model_name_or_path`

        Parameters
        ----------
        pretrained_model_name_or_path : str
            the pretrained model path

        Raises
        ------
        KeyError
            _description_
        """

        """ >> initialize vars <<"""

        model_type = None
        model_url = None
        server_name = None
        model_config = None
        model_archive = None
        backend = None
        is_local = False
        backend_default_flag = False
        image_processor_config_file_path = None

        extra_dict = OrderedDict()

        """ >> get model infomation <<"""
        if pretrained_model_name_or_path in MODEL_ARCHIVE_CONFIG:
            model_archive = pretrained_model_archive_parse(
                pretrained_model_name_or_path,
                MODEL_ARCHIVE_CONFIG[pretrained_model_name_or_path],
                region,
            )
            model_type = model_archive.get("type", None)
            model_url = model_archive.get("url_or_path", None)
            server_name = model_archive.get("server", None)
            is_local = False
        elif os.path.exists(pretrained_model_name_or_path) and os.path.isdir(pretrained_model_name_or_path):
            image_processor_config_file_path = file_exist(pretrained_model_name_or_path, IMAGE_PROCESSOR_CONFIG_NAMES)
            assert (
                image_processor_config_file_path is not None
            ), f"please check the image processor config file in {pretrained_model_name_or_path}"
            model_config_path = file_exist(pretrained_model_name_or_path, MODEL_CONFIG_NAMES)
            assert (
                model_config_path is not None
            ), f"please make sure the model config file exist in f{pretrained_model_name_or_path}"

            config_dict_ = file_read(model_config_path)

            model_type = config_dict_.get("model_type", None)
            is_local = True
        else:
            logger.warning("can not found model location, load from huggingface...")
        """ >> preprocessing: download files << """
        if server_name:
            if not image_processor_config_file_path:
                image_processor_config_file_path = cache_file(
                    pretrained_model_name_or_path,
                    IMAGE_PROCESSOR_CONFIG_NAMES,
                    model_url,
                    model_type,
                    if_cache=if_cache,
                    **kwargs,
                )

        """ >> load processor class <<"""

        image_processor = None

        extra_dict.update(
            {
                "server_name": server_name,
                "archive_name": pretrained_model_name_or_path,
                "model_type": model_type,
                "remote_url": model_url,
                "region": region,
            }
        )

        if not model_type:
            logger.info("try to use transformers to load image processor~")
            try:
                from transformers import AutoImageProcessor

                image_processor = AutoImageProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)
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
                from .image_processing_auto_hf import HFAutoImageProcessor

                pretrained_model_name_or_path_ = (
                    hf_name_or_path_check(
                        pretrained_model_name_or_path,
                        model_url,
                        model_type,
                    )
                    if not is_local
                    else pretrained_model_name_or_path
                )
                image_processor = HFAutoImageProcessor.from_pretrained(
                    pretrained_model_name_or_path_, *inputs, **kwargs
                )
            elif backend == "titan":
                # TODO (junwei.Dong): 支持特殊的titan模型
                raise NotImplementedError(backend)
            elif backend == "fex":
                # TODO (junwei.Dong): 支持特殊的fex模型
                raise NotImplementedError(backend)
            else:
                backend_default_flag = True

            if backend_default_flag:
                image_processor_name_tuple = MODELZOO_CONFIG[model_type]["image_processor"]
                (
                    image_processor_module_package,
                    image_processor_module_name,
                ) = MODELZOO_CONFIG.to_module(image_processor_name_tuple)
                image_processor_class = lazy_model_import(image_processor_module_package, image_processor_module_name)
                AutoHubClass.kwargs = extra_dict

                image_processor_config = file_read(image_processor_config_file_path)
                image_processor_config.update(kwargs)
                image_processor = image_processor_class(**image_processor_config)

        """ >> processsor post processing <<"""

        if image_processor:
            setattr(image_processor, "extra_dict", extra_dict)

        return image_processor
