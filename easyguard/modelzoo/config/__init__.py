import os
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Optional

MODEL_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "models.yaml")
MODEL_ARCHIVE_PATH_BACKUP = os.path.join(os.path.dirname(__file__), "archive.yaml")
TOS_FILES_PATH = os.path.join(os.path.dirname(__file__), "tos_files.yaml")
MODELZOO_NAME = "models"
YAML_DEEP = 3

import importlib

from ... import EASYGUARD_CONFIG_CACHE
from ...core.auto import EASYGUARD_PATH
from ...utils import HDFS_HUB_CN, YamlConfig, _LazyModule, hmget, load_yaml, logging

MODEL_ARCHIVE_PATH = os.path.join(EASYGUARD_CONFIG_CACHE, "archive.yaml")
MODEL_ARCHIVE_PATH_REMOTE = os.path.join(HDFS_HUB_CN, "config", "archive.yaml")
"""
config: tokenizer, vocab, model全都通过models.yaml来连接, 因此, 很多操作就可以借助models.yaml来进行简化,
例如:
模型注册: 直接将自主开发的模型一次注入到models.yaml文件里即可调用, 无需在auto各个模块进行配置
模型开发: 在模型的__init__函数里只需要利用typing.TYPE_CHECKING来辅助代码提示即可,无需手动lazyimport, 可参照deberta模型进行开发
模型懒加载: 不再需要各种mapping的存在, 因为models.yaml已经把各自模型的配置归类在一起了, 所以直接借助models.yaml即可轻松完成模块按需懒加载使用
"""

logger = logging.get_logger(__name__)


class ModelZooYaml(YamlConfig):
    @classmethod
    def to_module(cls, config: tuple) -> tuple:
        """

        Parameters
        ----------
        config : tuple
            specific config
            example:
            ('models.bert.configuration_bert.config', 'BertConfig')

        Returns
        -------
        str
            module name
            example:
                'models.bert.configuration_bert.BertConfig'
        """
        module_split = config[0].split(".")
        return ".".join(module_split[:-1]), config[-1]

    def check(self):
        """check modelzoo config yaml:
        1. the deepest level is `YAML_DEEP`.
        2. for a specific model, each key has an unique name.

        Returns
        -------

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        """
        global MODELZOO_NAME, YAML_DEEP
        leafs = {}
        prefix = MODELZOO_NAME

        def dfs_leafs(data: Dict[str, Any], deep: int, leafs: List[str], prefix: str):
            global YAML_DEEP
            deep_ = deep + 1
            for key_item, value in data.items():
                prefix_ = f"{prefix}.{key_item}"
                if isinstance(value, dict):
                    if deep_ > YAML_DEEP:
                        raise ValueError(f"the modelzoo config `{prefix_}` should not be a dict~")
                    dfs_leafs(value, deep_, leafs, f"{prefix_}")
                else:
                    leafs.append((prefix_, key_item))

        # for model_backend in self.config[MODELZOO_NAME].keys():
        for key_item, value in self.config[MODELZOO_NAME].items():
            leafs[key_item] = []
            dfs_leafs(value, 2, leafs[key_item], f"{prefix}.{key_item}")

        for key_item, value in leafs.items():
            paths, leaf_values = zip(*leafs[key_item])
            temp_dict = {}
            for index, leaf_value_item in enumerate(leaf_values):
                if leaf_value_item in temp_dict:
                    raise ValueError(
                        f"the `{paths[index]}` and `{temp_dict[leaf_value_item]}` have the same key `{leaf_value_item}`~"
                    )
                else:
                    temp_dict[leaf_value_item] = paths[index]

    def model_detail_config(self):
        """_summary_"""
        model_index = 1
        self.models = {}
        self.models_ = {}
        for leaf_item in self.leafs:
            prefix_, key_, value_ = leaf_item
            if prefix_.startswith("models."):
                model_ = prefix_.split(".")[model_index]

                if model_ in self.models:
                    self.models[model_][key_] = (prefix_, value_)
                    self.models_[model_][key_] = value_
                else:
                    self.models[model_] = {key_: (prefix_, value_)}
                    self.models_[model_] = {key_: value_}

    def get_mapping(self, *keys) -> OrderedDict:
        """get mappings for huggingface models

        example:

        Returns
        -------
        OrderedDict
            a specific mapping for target keys
        """

        if not hasattr(self, "models"):
            self.model_detail_config()

        mapping = {}

        for model_, config_ in self.models.items():
            values_ = []
            for key_ in keys:
                value_ = config_.get(key_, None)
                values_.append(value_[-1] if value_ is not None else None)
            if len(values_) > 1:
                mapping[model_] = tuple(values_)
            else:
                model_value_ = values_[0]
                if model_value_ is not None:
                    mapping[model_] = model_value_

        mapping_list = [(key_item, value) for key_item, value in mapping.items()]

        return OrderedDict(mapping_list)

    def initialize(self):
        self.check()
        self.model_detail_config()
        self.sys_register()

    def sys_register(self):
        """该函数基于models.yaml对easyguard里的所有模型实现了一次懒加载，代替了hf每个模型的__init__里的懒加载功能，
        即只要在models.yaml里对想要暴露的模块加以声明，那么在模型模块的__init__里只需要通过TYPE_CHECKING来进行伪加载【方便写代码的时候有提示】就行了，
        example:
        old version:
        >>> from easyguard.models.bert.configuration_bert import BertConfig
        now:
        >>> from easyguard.models.bert import BertConfig
        这样之后对于hf的支持就更加彻底了，在注入hf模型的时候，可以把__init__里除了TYPING_CHECKING以外的代码全部剔除
        """
        for model_name_, model_info_ in self.config["models"].items():
            import_structure_ = {}
            name_ = ".".join([EASYGUARD_PATH, model_name_])
            file_ = os.path.join(
                os.environ["EASYGUARD_HOME"],
                name_.replace(".", os.path.sep),
                "__init__.py",
            )
            for key_, value_ in model_info_.items():
                if isinstance(value_, dict):
                    import_structure_[key_] = list(value_.values())
            sys.modules[name_] = _LazyModule(name_, file_, import_structure_)
        ...

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return self.models_.get(key, default)

    def __getitem__(self, key: str):
        if not hasattr(self, "models"):
            self.model_detail_config()
        if self.models.get(key, None):
            return self.models[key]
        raise KeyError(key)


MODELZOO_CONFIG = ModelZooYaml.yaml_reader(MODEL_CONFIG_PATH)
# if os.path.exists(MODEL_ARCHIVE_PATH):
#     os.remove(MODEL_ARCHIVE_PATH)

# os.makedirs(EASYGUARD_CONFIG_CACHE, exist_ok=True)
# hmget([MODEL_ARCHIVE_PATH_REMOTE], EASYGUARD_CONFIG_CACHE)
# MODEL_ARCHIVE_PATH_ = MODEL_ARCHIVE_PATH_BACKUP
# if (
#     os.path.exists(MODEL_ARCHIVE_PATH)
#     and os.path.getsize(MODEL_ARCHIVE_PATH) != 0
# ):
#     MODEL_ARCHIVE_PATH_ = MODEL_ARCHIVE_PATH
# logger.info(f"the path of the loaded archive file: {MODEL_ARCHIVE_PATH_}")

# MODEL_ARCHIVE_CONFIG = load_yaml(MODEL_ARCHIVE_PATH_)
MODEL_ARCHIVE_CONFIG = load_yaml(MODEL_ARCHIVE_PATH_BACKUP)
MODELZOO_CONFIG.initialize()
