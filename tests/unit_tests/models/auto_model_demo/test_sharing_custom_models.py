import timm
from torch import nn
from transformers import PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING, MODEL_MAPPING

from easyguard.models.auto_model_demo import ResnetConfig, ResnetModelForImageClassification


def test_sharing_custom_models():
    resnet50d_config = ResnetConfig(block_type="bottleneck", stem_width=32, stem_type="deep", avg_down=True)
    assert isinstance(resnet50d_config, PretrainedConfig)
    resnet50d_config.save_pretrained("custom_resnet")

    resnet50d = ResnetModelForImageClassification(resnet50d_config)
    pretrained_model = timm.create_model("resnet50d", pretrained=True)
    resnet50d.model.load_state_dict(pretrained_model.state_dict())
    resnet50d.save_pretrained("custom_resnet")

    assert "resnet_hkt" in CONFIG_MAPPING.keys()
    assert ResnetConfig in MODEL_MAPPING.keys()
    assert ResnetConfig in MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.keys()

    resnet50d_config_bak = ResnetConfig.from_pretrained("custom_resnet")
    assert isinstance(resnet50d_config_bak, PretrainedConfig)

    resnet50d_bak = ResnetModelForImageClassification.from_pretrained("custom_resnet")
    assert isinstance(resnet50d_bak, nn.Module)
    assert isinstance(resnet50d_bak, ResnetModelForImageClassification)
