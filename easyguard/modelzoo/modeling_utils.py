from abc import ABC, abstractmethod
from modulefinder import Module
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from prettytable import PrettyTable
from torch import Tensor, nn

from ..utils import (  # DUMMY_INPUTS,; FLAX_WEIGHTS_NAME,; SAFE_WEIGHTS_INDEX_NAME,; SAFE_WEIGHTS_NAME,; TF2_WEIGHTS_NAME,; TF_WEIGHTS_NAME,
    load_pretrained_model_weights,
    logging,
)
from .hub import AutoHubClass

# from .utils.versions import require_version_core
# -*- coding: utf-8 -*-
logger = logging.get_logger(__name__)

# TODO (junwei.Dong): 一些基础的通用的方法和对应平台终端的方法可以在ModelBase里进行注册，自己写的模型尽量继承该基类
# EasyGuard Model base class


class ModelBase(nn.Module, AutoHubClass):
    def __init__(self, **kargs) -> None:
        super().__init__()

    def load_pretrained_weights(self, weight_file_path, **kwargs):
        # the default weights load function, we can overload this function in a specific model class and the `AutoModel.from_pretrained` function will call it after loading the architecture of the model
        load_pretrained_model_weights(self, weight_file_path, **kwargs)
