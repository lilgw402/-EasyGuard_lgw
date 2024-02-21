from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, Union

import numpy as np
import PIL
import torch

from ..utils import logging
from .hub import AutoHubClass

logger = logging.get_logger(__name__)

# image default setting
ImageInput = Union[
    "PIL.Image.Image",
    np.ndarray,
    "torch.Tensor",
    List["PIL.Image.Image"],
    List[np.ndarray],
    List["torch.Tensor"],
]  # noqa

# mean and std come from ImageNet dataset
MEAN_IMAGE = [0.485, 0.456, 0.406]
STD_IMAGE = [0.229, 0.224, 0.225]
RESIZE_IMAGE = 256
CENTER_IMAGE = 224


# for image preprocess
class ProcessorImageBase(ABC, AutoHubClass):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        if not getattr(self, "mean", None):
            self.mean = MEAN_IMAGE
        if not getattr(self, "std", None):
            self.std = STD_IMAGE

    def __call__(self, image, **kwds: Any) -> Any:
        return self.preprocess(image, **kwds)

    @abstractmethod
    def preprocess(self, image, **kwds: Any):
        """for image data preprocessing"""


# for auto preprocess
class ProcessorBase(AutoHubClass):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        if not getattr(self, "mean", None):
            self.mean = MEAN_IMAGE
        if not getattr(self, "std", None):
            self.std = STD_IMAGE

    def __call__(self, text=None, image=None, **kwds: Any) -> Any:
        return self.preprocess(text=text, image=image, **kwds)

    def text_process(self, text, **kwargs) -> Dict[str, Any]:
        """preprocess for text"""
        return self.tokenizer(text)

    def image_process(self, image, **kwargs):
        """preprocess for image"""
        return self.image_processor(image)

    def preprocess(self, text=None, image=None, **kwds):
        """for image and text data processing"""

        if not text and not image:
            raise ValueError("You have to specify either text or images. Both cannot be none.")

        if text:
            encoding = self.text_process(text)
        if image:
            image_feature = self.image_process(image)

        if text and image:
            assert isinstance(encoding, dict), f"`encoding` should be a {dict}"
            encoding["pixel_values"] = image_feature
            return encoding
        elif text:
            return encoding
        else:
            return OrderedDict(pixel_values=image_feature)
