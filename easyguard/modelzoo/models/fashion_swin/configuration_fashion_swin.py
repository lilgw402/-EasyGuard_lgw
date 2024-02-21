from dataclasses import dataclass
from typing import List

from ...configuration_utils import ConfigBase


@dataclass
class FashionSwinConfig(ConfigBase):
    model_name: str
    model_type: str
    img_size: int
    patch_size: int
    in_chans: int
    num_classes: int
    embed_dim: int
    depths: List[int]
    num_heads: List[int]
    window_size: int
    mlp_ratio: float
    qkv_bias: bool
    # qk_scale: Optional[float]
    drop_rate: float
    attn_drop_rate: float
    drop_path_rate: float
    ape: bool
    patch_norm: bool
    use_checkpoint: bool

    def __post_init__(self):
        super().__init__(**self.__dict__)

    def config_update_for_pretrained(self, **kwargs):
        ...
