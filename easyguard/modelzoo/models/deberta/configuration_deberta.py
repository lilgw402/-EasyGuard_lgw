from dataclasses import dataclass
from typing import Any, Dict

from ....utils import get_configs, logging
from ...configuration_utils import ConfigBase

"""Deberta configuration"""

logger = logging.get_logger(__name__)


@dataclass
class DeBERTaConfig(ConfigBase):
    model_name: str
    model_type: str
    vocab_size: int
    dim: int
    dim_ff: int
    dim_shrink: int
    n_segments: int
    max_len: int
    n_heads: int
    n_layers: int
    embedding_dim: int
    act: str
    layer_norm_eps: float
    pool: bool
    p_drop_hidden: float
    p_drop_attn: float
    embedding_dropout: float
    padding_index: int
    attention_clamp_inf: bool
    ignore_index: int
    initializer_range: float
    calc_mlm_accuracy: bool
    omit_other_output: bool
    layernorm_fp16: bool
    max_relative_positions: int
    abs_pos_embedding: bool
    tie_embedding: bool
    use_emd: bool
    num_emd_groups: int
    emd_group_repeat: int
    layernorm_type: str
    head_layernorm_type: str
    extra_da_transformer_config: Dict[str, Any]

    def __post_init__(self):
        super().__init__(**self.__dict__)

    def config_update_for_pretrained(self, **kwargs):
        """for pretrained model"""
        _, model_config = get_configs(**kwargs)
        if model_config["features_only"]:
            raise RuntimeError("features_only is not supported for bert model.")
        model_config.pop("features_only")

        editable_keys = set(
            [
                "p_drop_hidden",
                "p_drop_attn",
                "calc_mlm_accuracy",
                "omit_other_output",
                "layernorm_fp16",
                "omit_other_attn_output",
                "use_emd",
                "dim_shrink",
            ]
        )
        for k, v in model_config.items():
            if k == "extra_da_transformer_config":
                for kk, vv in model_config[k].items():
                    if kk in editable_keys:
                        self.__dict__[k][kk] = vv
                    else:
                        raise ValueError(f"Cannot edit 'extra_da_transformer_config.{kk}={vv}' with pretrained=True")
            elif k in editable_keys:
                self.__dict__[k] = v
