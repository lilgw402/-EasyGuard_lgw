# 关于如何在AutoModel注册一个新模型，从而可以使用AutoModel.from_pretrained直接调用模型，可以参考文档：
# https://huggingface.co/docs/transformers/custom_models

from easyguard.models.auto import AutoConfig, AutoModel, AutoModelForImageClassification

from .configuration_resnet import ResnetConfig
from .modeling_resnet import ResnetModel, ResnetModelForImageClassification

__all__ = [
    "ResnetConfig",
    "ResnetModel",
    "ResnetModelForImageClassification",
]


AutoConfig.register("resnet_hkt", ResnetConfig)
AutoModel.register(ResnetConfig, ResnetModel)
AutoModelForImageClassification.register(ResnetConfig, ResnetModelForImageClassification)
