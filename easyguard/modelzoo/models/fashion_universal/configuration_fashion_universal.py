from dataclasses import dataclass

from ...configuration_utils import ConfigBase


@dataclass
class FashionUniversalConfig(ConfigBase):
    model_name: str
    model_type: str
    input_size: int
    patch_size: int
    in_channels: int
    dim: int
    embedding_size: int
    depths: int
    num_heads: int
    mlp_ratio: int
    # qkv_bias: bool
    # qk_scale: Optional[float]
    # drop_rate: float
    # attn_drop_rate: float
    drop_path_rate: float
    use_checkpoint: bool

    def __post_init__(self):
        super().__init__(**self.__dict__)

    def config_update_for_pretrained(self, **kwargs):
        ...
