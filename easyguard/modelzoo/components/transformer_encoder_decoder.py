"""Transformer encoder/decoder implementations from ptx"""
import math
import os
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import logging
from .activation import Activation
from .layernorm import LayerNormTypes

logger = logging.get_logger(__name__)
try:
    from torch.fx import wrap as fx_wrap
except:
    logger.warning("Failed to import torch.fx.wrap")

    def fx_wrap(func):
        return func


# veGiantModel layers
try:
    import veGiantModel
    from veGiantModel.module import (
        ColumnParallelLinear,
        ColumnParallelLinearTranspose,
        ColumnSerialLinear,
        ColumnSerialLinearTranspose,
        MockModule,
        RowParallelLinear,
        RowSerialLinear,
    )
except:
    veGiantModel = None
    ColumnParallelLinear = None
    RowParallelLinear = None
    ColumnSerialLinear = None
    RowSerialLinear = None
    ColumnParallelLinearTranspose = None
    ColumnSerialLinearTranspose = None
    MockModule = None

# FT ops will be enabled through fx, here all ops are set to None
FTLinear = None
FTTranspose = None
FTTransposeV1 = None
FTMatMul = None
FTLinearTranspose = None
FTAttention = None
FTFusedAttention = None
FTSoftmax = None
FTLayerNorm = None


MHAOutput = namedtuple(
    "MHAOutput",
    [
        "attn_outputs",
        "attn_probs",
        "attn_weights",
        "q_states",
        "k_states",
        "v_states",
        "attn_bias",
    ],
)

TransformerEncoderLayerOutput = namedtuple(
    "TransformerEncoderLayerOutput",
    [
        "hidden_states",
        "attn_probs",
        "attn_weights",
        "q_states",
        "k_states",
        "v_states",
        "attn_bias",
    ],
)

TransformerDecoderLayerOutput = namedtuple(
    "TransformerDecoderLayerOutput",
    [
        "hidden_states",
        "self_attn_weights",
        "cross_attn_weights",
        "present_key_value",
        "attn_bias",
        "encoder_attn_bias",
    ],
)

TransformerEncoderOutput = namedtuple(
    "TransformerEncoderOutput",
    [
        "last_hidden_state",
        "all_hidden_states",
        "all_attn_probs",
        "all_attn_weights",
        "all_q_states",
        "all_k_states",
        "all_v_states",
    ],
)

TransformerDecoderOutput = namedtuple(
    "TransformerDecoderOutput",
    [
        "last_hidden_state",
        "all_hidden_states",
        "all_self_attns",
        "all_cross_attns",
        "all_key_values",
    ],
)

Seq2SeqModelOutput = namedtuple(
    "Seq2SeqModelOutput",
    [
        "last_hidden_state",
        "past_key_values",
        "decoder_hidden_states",
        "decoder_attentions",
        "cross_attentions",
        "encoder_last_hidden_state",
        "encoder_hidden_states",
        "encoder_attentions",
    ],
)


@dataclass
class TransformerConfig:
    n_layers: int
    dim: int
    n_heads: int
    dim_ff: int
    act: str = "gelu"
    layernorm_type: str = "v0"  # See `ptx.ops.layernorm`
    use_pre_layernorm: bool = False
    use_realformer: bool = False
    p_drop_hidden: float = 0.1
    p_drop_hidden2: Optional[float] = None
    p_drop_attn: float = 0.1
    return_layers: List[int] = field(default_factory=list)
    clamp_inf_nan: bool = False
    layer_norm_eps: float = 1e-5
    max_batch_size: int = 128
    max_seq_length: int = 512
    fp16: bool = True
    layernorm_fp16: bool = False
    remove_padding: bool = False
    fuse_qkv_projs: bool = False
    omit_other_attn_output: bool = False
    layer_grad_checkpoint: bool = False
    decoder_layer_grad_checkpoint: bool = False
    use_ft_softmax: bool = False
    disable_ft_softmax_dropout: bool = False
    use_ft_layernorm: bool = False
    use_apex_mha_mask_additive: bool = False
    use_ft_linear_in_attn: bool = False
    use_mp_linear_in_attn: bool = False
    use_ft_transpose_in_attn: bool = False
    use_ft_mm_in_attn: bool = False
    use_ft_mm_in_attn_wo_scale: bool = False
    use_ft_linear_in_attn_out: bool = False
    use_mp_linear_in_attn_out: bool = False
    use_ft_linear_in_ffn: bool = False
    use_mp_linear_in_ffn: bool = False
    mha_acts_unite_d01: bool = True
    dropout_in_ffn: bool = False
    use_ft_ffn_linear_fusion: bool = False
    use_ffn_output_dropout: bool = False
    use_ft_attn_out_proj_dropout_fusion: bool = False
    use_ft_linear_transpose_fusion_in_attn: bool = False
    use_ft_remove_pad: bool = False
    use_ft_fused_attn: bool = False
    n_decoder_layers: int = 0
    return_decoder_layers: List[int] = field(default_factory=list)
    pad_seq_len_even: bool = False
    use_moe: bool = False
    use_moe_type: str = "pwff"
    use_moe_transformer_layer: str = ""
    use_moe_decoder_transformer_layer: str = ""
    moe_k: int = 1
    moe_experts: int = 8
    moe_output_dropout_prob: float = 0.0
    moe_min_capacity: int = 1
    moe_capacity_factor: float = 1.0
    moe_l_aux_factor: float = 1.0
    moe_z_loss_factor: float = 0.0
    moe_noisy_gate_policy: Optional[str] = None
    moe_eval_capacity_factor: float = 1.0
    moe_experts_decay: bool = False
    moe_dim: int = 1024
    moe_dropout: bool = False
    gate_bias: bool = False
    moe_flexible_validate: bool = True
    moe_drop_token: bool = False
    moe_random_token_select: bool = False
    moe_load_balanced: bool = False
    moe_enable_token_drop: bool = False
    moe_warmup_stage: Optional[str] = None
    moe_warmup_steps: str = ""
    moe_expert_shape: str = "abc->abd"
    use_moe_attn: bool = False
    use_moe_transformer_layer_attn: str = ""
    moe_k_attn: int = 2
    moe_experts_attn: int = 32
    moe_dropout_attn: bool = False
    moe_l_aux_factor_attn: float = 0.01
    moe_dim_attn: int = 1024
    moe_load_balanced_attn: bool = False
    moe_attn_expert_shape: str = "abc->abd"
    use_moe_lego: bool = False
    pos_emb_type: str = ""
    use_ft_preset: str = ""
    n_t5_rel_pos_buckets: int = 32
    beit_rel_pos_window_size: List[int] = field(default_factory=list)
    use_deep_norm: bool = False
    deep_norm_enc_alpha: float = 1.0
    deep_norm_enc_beta: float = 1.0
    layer_grad_checkpoint_skipped_per_blocks: int = 1

    def __post_init__(self):
        if self.p_drop_hidden2 is None:
            self.p_drop_hidden2 = self.p_drop_hidden
        self.head_dim = self.dim // self.n_heads
        assert self.dim % self.n_heads == 0, f"`dim` must be divisible by `n_heads` (got {self.dim}/{self.n_heads})."

        if self.use_ft_preset != "":
            raise NotImplementedError("use_ft_preset is not supported in titan")
            # ft_preset = FT_PRESET[self.use_ft_preset]
            # for k, v in ft_preset.items():
            #     if hasattr(self, k):
            #         setattr(self, k, v)

        if self.use_deep_norm:
            self.deep_norm_enc_alpha = (2 * self.n_layers) ** 0.25
            self.deep_norm_enc_beta = (8 * self.n_layers) ** -0.25


Config = TransformerConfig


@torch.jit.script
def hack_torch_trace(a: torch.Tensor, b: Optional[torch.Tensor] = None) -> torch.Tensor:
    if b is None:
        return a
    return a + b


def _expand_mask(
    mask: torch.Tensor,
    dtype: torch.dtype,
    tgt_len: Optional[int] = None,
    mask_value: Optional[int] = None,
) -> torch.Tensor:
    """
    Expands attn_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    if mask.dim() == 4:
        return mask
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    if mask_value is None:
        mask_value = torch.finfo(dtype).min
    return inverted_mask.masked_fill(inverted_mask.bool(), mask_value)


def rescue_inf_nan(x: torch.Tensor) -> torch.Tensor:
    # if torch.isinf(x).any() or torch.isnan(x).any():
    #     clamp_value = torch.finfo(x.dtype).max - 1000
    #     x = torch.clamp(x, min=-clamp_value, max=clamp_value)
    x = torch.nan_to_num(x)
    return x


@fx_wrap
def _fx_multi_head_attention_mask_score(mask, scores):
    if mask is not None:
        if mask.dim() == 2:
            mask = mask[:, None, None, :]
            scores -= 10000.0 * (1.0 - mask)
        else:
            scores += mask
    return mask, scores


class BareMultiHeadAttention(nn.Module):
    def __init__(self, config: TransformerConfig, is_decoder: bool = False, **kwargs):
        super().__init__()
        if isinstance(config, dict):
            config = TransformerConfig(**config)
        self.config = config
        self.is_decoder = is_decoder

        self.proj_q = nn.Linear(config.dim, config.dim)
        self.proj_k = nn.Linear(config.dim, config.dim)
        self.proj_v = nn.Linear(config.dim, config.dim)
        self.dropout = nn.Dropout(config.p_drop_attn)
        self.n_heads = config.n_heads
        self.head_size = config.head_dim
        self.all_head_size = self.n_heads * self.head_size

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        """
        new_x_shape = x.size()[:-1] + (self.n_heads, self.head_size)
        # Tensor.view supports tuple inputs
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attn_mask=None,
        q_state=None,
        kv_state=None,
        past_kv=None,
        attn_bias=None,
    ) -> torch.Tensor:
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        x = hidden_states
        mask = attn_mask

        is_x_attn = kv_state is not None

        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q = self.proj_q(x if q_state is None else q_state)
        q = self.transpose_for_scores(q)

        if is_x_attn:
            if past_kv is not None:
                k = past_kv[0]
                v = past_kv[1]
            else:
                k = self.transpose_for_scores(self.proj_k(kv_state))
                v = self.transpose_for_scores(self.proj_v(kv_state))
        else:
            k = self.transpose_for_scores(self.proj_k(x))
            v = self.transpose_for_scores(self.proj_v(x))
            if past_kv is not None:
                k = torch.cat([past_kv[0], k], dim=2)
                v = torch.cat([past_kv[1], v], dim=2)

        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_size**0.5)

        if attn_bias is not None:
            scores += attn_bias
        mask, scores = _fx_multi_head_attention_mask_score(mask, scores)

        probs = self.dropout(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous()
        # -merge-> (B, S, D)
        new_context_layer_shape = h.size()[:-2] + (self.all_head_size,)
        h = h.view(new_context_layer_shape)

        if not self.is_decoder:
            return h
        return h, (k, v)


class BareTransformerEncoderLayer(nn.Module):
    attention_class = BareMultiHeadAttention

    def __init__(self, config: TransformerConfig, order: int = -1):
        super().__init__()
        if isinstance(config, dict):
            config = TransformerConfig(**config)
        self.config = config
        self.dim = self.config.dim
        self.l_aux = -1

        self.attn = self.attention_class(config)
        self.proj = nn.Linear(config.dim, config.dim)
        self.norm1 = LayerNormTypes[config.layernorm_type](config.dim, config.layer_norm_eps)
        self.pwff = PositionWiseFeedForward(config)
        if (
            config.use_moe
            and "pwff" in config.use_moe_type.split(",")
            and str(order) in config.use_moe_transformer_layer.split(",")
        ):
            import janus.layer

            self._use_moe = True
            self.pwff = janus.layer.MoE(
                hidden_size=config.moe_dim,
                expert=self.pwff,
                num_experts=config.moe_experts,
                k=config.moe_k,
                capacity_factor=config.moe_capacity_factor,
                eval_capacity_factor=config.moe_eval_capacity_factor,
                output_dropout_prob=config.moe_output_dropout_prob,
                min_capacity=config.moe_min_capacity,
                noisy_gate_policy=config.moe_noisy_gate_policy,
                is_dropout=config.moe_dropout,
                flexible_validate=config.moe_flexible_validate,
                gate_bias=config.gate_bias,
            )
        else:
            self._use_moe = False
        self.norm2 = LayerNormTypes[config.layernorm_type](config.dim, config.layer_norm_eps)
        self.dropout1 = nn.Dropout(config.p_drop_hidden)
        self.dropout2 = nn.Dropout(config.p_drop_hidden2)

        if config.layernorm_fp16:
            self.norm1._simply_cast = True
            self.norm2._simply_cast = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor,
        q_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Multi-Head Self-Attention
        residual = hidden_states if q_state is None else q_state
        self_attn_output = self.attn(hidden_states, attn_mask=attn_mask, q_state=q_state)
        hidden_states = self.proj(self_attn_output)
        hidden_states = self.dropout1(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.norm1(hidden_states)

        # Position-wise Feed-Forward Networks
        residual = hidden_states
        if self._use_moe:
            hidden_states, l_aux, _ = self.pwff(hidden_states)
            if self.training:
                self.l_aux = l_aux
        else:
            hidden_states = self.pwff(hidden_states)
        hidden_states = self.dropout2(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.norm2(hidden_states)

        if self.config.clamp_inf_nan and (hidden_states.dtype == torch.float16):
            hidden_states = rescue_inf_nan(hidden_states)

        return hidden_states


class BareTransformerEncoder(nn.Module):
    layer_class = BareTransformerEncoderLayer

    def __init__(self, config: TransformerConfig):
        super().__init__()
        if isinstance(config, dict):
            config = TransformerConfig(**config)
        self.config = config
        self.dim = self.config.dim
        self.blocks = nn.ModuleList([self.layer_class(config, order=i) for i in range(config.n_layers)])
        self.l_aux = list()

    def forward(self, hidden_states, attn_mask=None, use_namedtuple_output=False) -> List[torch.Tensor]:
        """
        Args:
            hidden_states: shape (bsz, seq_len, dim); embeddings
            attn_mask: shape (bsz, seq_len); Mask to avoid performing attention on padding token indices;
        """
        attn_mask_expanded = _expand_mask(attn_mask, hidden_states.dtype) if attn_mask.dim() != 4 else attn_mask

        all_hidden_states = [hidden_states]

        l_aux = list()
        for block in self.blocks:
            hidden_states = block(hidden_states, attn_mask_expanded)
            all_hidden_states.append(hidden_states)
            if self.training:
                if getattr(block, "_use_moe", False):
                    l_aux.append(block.l_aux)
        if self.training:
            self.l_aux = l_aux

        if not use_namedtuple_output:
            return all_hidden_states
        return TransformerEncoderOutput(
            last_hidden_state=hidden_states,
            all_hidden_states=all_hidden_states,
            all_attn_probs=[],
            all_attn_weights=[],
            all_q_states=[],
            all_k_states=[],
            all_v_states=[],
        )


class PositionWiseFeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        if self.config.use_ft_ffn_linear_fusion:
            assert config.act == "gelu"
            assert self.config.use_ft_linear_in_ffn

        if self.config.use_ft_linear_in_ffn:
            assert FTLinear is not None
            fc1_cls = fc2_cls = FTLinear
            fc1_kwargs = {}
            fc2_kwargs = {}
            if self.config.use_mp_linear_in_ffn:
                assert ColumnParallelLinear is not None
                assert RowParallelLinear is not None
                fc1_cls = ColumnParallelLinear
                fc2_cls = RowParallelLinear
                fc1_kwargs["use_ft"] = True
                fc2_kwargs["use_ft"] = True
            if self.config.use_ft_ffn_linear_fusion:
                fc1_dropout = 0.0 if not self.config.dropout_in_ffn else config.p_drop_hidden
                fc2_dropout = 0.0 if not self.config.use_ffn_output_dropout else config.p_drop_hidden
                fc1_kwargs["act_gelu"] = True
                fc1_kwargs["dropout_rate"] = fc1_dropout
                fc2_kwargs["dropout_rate"] = fc2_dropout
                self.fc1 = fc1_cls(config.dim, config.dim_ff, **fc1_kwargs)
                self.fc2 = fc2_cls(config.dim_ff, config.dim, **fc2_kwargs)
            else:
                self.fc1 = fc1_cls(config.dim, config.dim_ff, **fc1_kwargs)
                self.fc2 = fc2_cls(config.dim_ff, config.dim, **fc2_kwargs)
        elif self.config.use_mp_linear_in_ffn:
            assert ColumnParallelLinear is not None
            assert RowParallelLinear is not None
            self.fc1 = ColumnParallelLinear(config.dim, config.dim_ff)
            self.fc2 = RowParallelLinear(config.dim_ff, config.dim)
        else:
            self.fc1 = nn.Linear(config.dim, config.dim_ff)
            self.fc2 = nn.Linear(config.dim_ff, config.dim)
        self.act = Activation(config.act)
        self.dropout = nn.Dropout(config.p_drop_hidden)  # TODO: Is this dropout redundant?

    def forward(self, x) -> torch.Tensor:
        # (bsz, seq_len, dim) -> (bsz, seq_len, dim_ff / model_parallel_size) -> (bsz, seq_len, dim)
        if not self.config.use_ft_ffn_linear_fusion:
            fc1_out = self.act(self.fc1(x))
            if self.config.dropout_in_ffn:
                fc1_out = self.dropout(fc1_out)
            fc2_out = self.fc2(fc1_out)
            if self.config.use_ffn_output_dropout:
                fc2_out = self.dropout(fc2_out)
            return fc2_out
        else:
            fc1_out = self.fc1(x)
            fc2_out = self.fc2(fc1_out)
            return fc2_out


class AttentionExpertFTLinearTranspose(nn.Module):
    def __init__(self, dim, dim2, n_heads):
        super().__init__()
        self.proj_k = FTLinearTranspose(dim, dim2, n_heads)
        self.proj_v = FTLinearTranspose(dim, dim2, n_heads)

    def forward(self, x):
        k_states = self.proj_k(x)
        v_states = self.proj_v(x)
        return torch.concat([k_states, v_states], dim=3)


class AttentionExpert(nn.Module):
    def __init__(self, dim, dim2, LinearCls):
        super().__init__()
        self.proj_k = LinearCls(dim, dim2)
        self.proj_v = LinearCls(dim, dim2)

    def forward(self, x):
        k_states = self.proj_k(x)
        v_states = self.proj_v(x)
        return torch.cat([k_states, v_states], dim=2)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention from 'Attention Is All You Need' paper
    https://arxiv.org/abs/1706.03762
    """

    def __init__(self, config: Config, is_decoder: bool = False, order: int = -1):
        super().__init__()
        if isinstance(config, dict):
            config = Config(**config)
        self.config = config
        self.is_decoder = is_decoder

        LinearCls = nn.Linear
        self.model_parallel_size = 1
        if self.config.use_mp_linear_in_attn:
            assert veGiantModel is not None, "Unable to import veGiantModel"
            self.model_parallel_size = veGiantModel.distributed.get_model_parallel_world_size()
        self._use_ft_linear_transpose_fusion = (
            self.config.use_ft_linear_transpose_fusion_in_attn and not self.config.use_ft_fused_attn
        )
        if self._use_ft_linear_transpose_fusion:
            assert not self.config.fuse_qkv_projs, "FasterLinearTranspose does not support `fuse_qkv_projs`"
            assert not self.config.remove_padding, "FasterLinearTranspose does not support `remove_padding`"
            assert FTLinearTranspose is not None
            LinearCls = FTLinearTranspose
            if self.config.use_mp_linear_in_attn:
                assert ColumnParallelLinearTranspose is not None
                LinearCls = ColumnParallelLinearTranspose
        elif self.config.use_ft_linear_in_attn:
            assert FTLinear is not None
            assert (
                not self.config.use_mp_linear_in_attn
            ), "ColumnParallelLinear does not support `use_mp_linear_in_attn`"
            LinearCls = FTLinear
        elif self.config.use_mp_linear_in_attn:
            assert ColumnParallelLinear is not None
            LinearCls = ColumnParallelLinear
        if self.config.use_ft_transpose_in_attn:
            # assert FTTranspose is not None
            assert FTTransposeV1 is not None
            self.faster_transpose = FTTransposeV1()
        else:
            self.faster_transpose = None
        if self.config.use_ft_mm_in_attn:
            assert FTMatMul is not None
            self.faster_matmul = FTMatMul()
        else:
            self.faster_matmul = None

        self.l_aux = 0
        self.use_moe = False
        if not config.fuse_qkv_projs:
            if self._use_ft_linear_transpose_fusion:
                if LinearCls is ColumnParallelLinearTranspose:
                    self.proj_k = LinearCls(config.dim, config.dim, config.n_heads, use_ft=True)
                    self.proj_v = LinearCls(config.dim, config.dim, config.n_heads, use_ft=True)
                    self.proj_q = LinearCls(config.dim, config.dim, config.n_heads, use_ft=True)
                else:
                    if self.config.use_moe_attn and str(order) in self.config.use_moe_transformer_layer_attn.split(
                        ","
                    ):
                        import janus.groups
                        import janus.layer

                        if janus.groups.is_initialized() and janus.groups.get_ep_size() > 1:
                            assert (
                                config.moe_experts_attn % janus.groups.get_ep_size() == 0
                            ), "num_expert must divide moe_ep_size"
                        self.use_moe = True
                        self.proj_moe = janus.layer.MoE(
                            hidden_size=config.moe_dim_attn,
                            expert=AttentionExpertFTLinearTranspose(config.dim, config.dim, config.n_heads),
                            num_experts=config.moe_experts_attn,
                            k=config.moe_k_attn,
                            noisy_gate_policy="None",
                            load_balanced=config.moe_load_balanced_attn,
                            enable_token_drop=False,
                            expert_shape=config.moe_attn_expert_shape,
                        )
                    else:
                        self.proj_k = LinearCls(config.dim, config.dim, config.n_heads)
                        self.proj_v = LinearCls(config.dim, config.dim, config.n_heads)
                    self.proj_q = LinearCls(config.dim, config.dim, config.n_heads)
            else:
                if self.config.use_moe_attn and str(order) in self.config.use_moe_transformer_layer_attn.split(","):
                    import janus.groups
                    import janus.layer

                    if janus.groups.is_initialized() and janus.groups.get_ep_size() > 1:
                        assert (
                            config.moe_experts_attn % janus.groups.get_ep_size() == 0
                        ), "num_expert must divide moe_ep_size"
                    self.use_moe = True
                    self.proj_moe = janus.layer.MoE(
                        hidden_size=config.moe_dim_attn,
                        expert=AttentionExpert(config.dim, config.dim, LinearCls),
                        num_experts=config.moe_experts_attn,
                        k=config.moe_k_attn,
                        noisy_gate_policy="None",
                        load_balanced=config.moe_load_balanced_attn,
                        enable_token_drop=False,
                        expert_shape=config.moe_attn_expert_shape,
                    )
                else:
                    self.proj_k = LinearCls(config.dim, config.dim)
                    self.proj_v = LinearCls(config.dim, config.dim)
                self.proj_q = LinearCls(config.dim, config.dim)
            self.proj_qkv = None
        else:
            assert not self.config.use_mp_linear_in_attn, "ColumnParallelLinear does not support `fuse_qkv_projs`"
            self.proj_qkv = LinearCls(config.dim, config.dim * 3)
            self.proj_k, self.proj_v, self.proj_q = None, None, None
        # in mp_linear mode, the num of heads & dim require adjustments
        self._use_mp_linear = self.config.use_mp_linear_in_attn and not issubclass(ColumnParallelLinear, MockModule)
        self.dropout = nn.Dropout(config.p_drop_attn)
        self.score_scale = config.head_dim**-0.5

        if self.config.use_ft_softmax:
            assert not self.config.use_realformer, "FasterSoftmax does not support `use_realformer`"
            # assert not self.config.clamp_inf_nan, 'FasterSoftmax does not support `clamp_inf_nan`'
            if self.config.omit_other_attn_output:
                logger.warning("FasterSoftmax does not return `attn_weights`")
            assert FTSoftmax is not None
            self.faster_softmax = FTSoftmax()

        self._use_ft_fused_attn = self.config.use_ft_fused_attn and not self.is_decoder  # TODO
        if self._use_ft_fused_attn:
            assert not self.config.use_realformer, "FasterFusedAttention does not support `use_realformer`"
            assert not self.config.fuse_qkv_projs, "FasterFusedAttention does not support `fuse_qkv_projs`"
            assert not self.config.remove_padding, "FasterFusedAttention does not support `remove_padding`"
            assert not self.config.mha_acts_unite_d01, "FasterFusedAttention does not support `mha_acts_unite_d01`"
            assert not self.config.use_mp_linear_in_attn, "ColumnParallelLinear does not support `use_ft_fused_attn`"
            if self.config.omit_other_attn_output:
                logger.warning("FasterFusedAttention does not return `attn_weights`")
            self.faster_attn = FTFusedAttention(config.n_heads, dropout_rate=config.p_drop_attn)
        else:
            self.faster_attn = None

        if self.config.pos_emb_type == "roformer":
            position_enc = np.array(
                [
                    [pos / np.power(10000, 2 * (j // 2) / self.config.head_dim) for j in range(self.config.head_dim)]
                    for pos in range(1024)
                ]
            )
            sin_pos = torch.repeat_interleave(torch.FloatTensor(np.sin(position_enc[:, 0::2])), 2, dim=-1)
            cos_pos = torch.repeat_interleave(torch.FloatTensor(np.cos(position_enc[:, 1::2])), 2, dim=-1)
            self.register_buffer("sin_pos", sin_pos[None, None, :, :])
            self.register_buffer("cos_pos", cos_pos[None, None, :, :])

    def _shape(self, tensor: torch.Tensor, bsz: int) -> torch.Tensor:
        """(bsz, seq_len, dim) -> (bsz, n_heads / model_parallel_size, seq_len, head_dim)"""
        n_heads = self.config.n_heads
        if self._use_mp_linear:
            n_heads = n_heads // self.model_parallel_size
        split_tensor = tensor.view(bsz, -1, n_heads, self.config.head_dim)
        if self.config.use_ft_transpose_in_attn:
            # return self.faster_transpose(split_tensor, *split_tensor.size())
            return self.faster_transpose(split_tensor)
        else:
            return split_tensor.permute(0, 2, 1, 3).contiguous()

    def _shape2(self, tensor: torch.Tensor, bsz: int) -> torch.Tensor:
        """(bsz * n_heads / model_parallel_size, seq_len, head_dim) -> (bsz, seq_len, dim)"""
        n_heads, dim = self.config.n_heads, self.config.dim
        if self._use_mp_linear:
            n_heads = n_heads // self.model_parallel_size
            dim = dim // self.model_parallel_size
        if self.config.mha_acts_unite_d01:
            tensor = tensor.view(bsz, n_heads, -1, self.config.head_dim)
        if self.config.use_ft_transpose_in_attn:
            # return self.faster_transpose(tensor, *tensor.size()).view(bsz, -1, self.config.dim)
            return self.faster_transpose(tensor).view(bsz, -1, dim)
        else:
            return tensor.permute(0, 2, 1, 3).reshape(bsz, -1, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        prev_attn_weights: Optional[torch.Tensor] = None,
        gathered_mask_index: Optional[torch.Tensor] = None,
        gathered_hidden_states: Optional[torch.Tensor] = None,
        q_state: Optional[torch.Tensor] = None,  # Only for Deberta so far
        attn_bias: Optional[torch.Tensor] = None,  # e.g. T5-style rel pos attn bias
    ) -> MHAOutput:
        """
        Args:
            hidden_states: shape (bsz, seq_len, dim)
            attn_mask: optional; shape (bsz, 1, tgt_len, src_len)
            key_value_states: optional; if provided, this layer is used as a cross-attention layer for decoder
            past_key_value: optional; tuple of (past key states, past value states)
            prev_attn_weights: optional; use realformer if provided
        """
        bsz, tgt_len, dim = hidden_states.size()
        if q_state is not None:  # TODO: ensure
            tgt_len = q_state.size(1)

        original_hidden_states = hidden_states
        remove_padding = gathered_hidden_states is not None
        if remove_padding:
            assert q_state is None, "Remove padding must not accept special query states"
            hidden_states = gathered_hidden_states

        is_cross_attention = key_value_states is not None

        shaped_projected_q, shaped_projected_k, shaped_projected_v = (
            None,
            None,
            None,
        )
        if self.config.fuse_qkv_projs:
            assert q_state is None, "Fused QKV must not accept special query states"
            assert not is_cross_attention, "Fused QKV must not be used with cross attention"
            assert not self.config.remove_padding, "Fused QKV can not be used with `remove_padding` now"
            assert not self.config.use_mp_linear_in_attn, "Fused QKV with model parallelism is not implemented yet"
            projected_qkv = self.proj_qkv(hidden_states)  # (bsz, seq_len, dim * 3)
            shaped_projected_qkv = (
                projected_qkv.view(bsz, -1, self.config.n_heads * 3, self.config.head_dim)
                .view(bsz, -1, 3, self.config.n_heads, self.config.head_dim)
                .permute(2, 0, 3, 1, 4)
                .contiguous()
                .view(3 * bsz * self.config.n_heads, -1, self.config.head_dim)
            )
            (
                shaped_projected_q,
                shaped_projected_k,
                shaped_projected_v,
            ) = shaped_projected_qkv.chunk(
                3, dim=0
            )  # (bsz * n_heads, seq_len, head_dim)
            if not self.config.mha_acts_unite_d01:
                shaped_projected_q = shaped_projected_q.view(bsz, self.config.n_heads, -1, self.config.head_dim)
                shaped_projected_k = shaped_projected_k.view(bsz, self.config.n_heads, -1, self.config.head_dim)
                shaped_projected_v = shaped_projected_v.view(bsz, self.config.n_heads, -1, self.config.head_dim)

        if not remove_padding:
            if self._use_ft_linear_transpose_fusion:
                q_states = self.proj_q(hidden_states if q_state is None else q_state)
            elif self._use_ft_fused_attn:
                q_states = self.proj_q(hidden_states if q_state is None else q_state)
            else:
                q_states = (
                    self._shape(
                        self.proj_q(hidden_states if q_state is None else q_state),
                        bsz,
                    )
                    if shaped_projected_q is None
                    else shaped_projected_q
                )  # (bsz, n_heads, seq_len, head_dim)
        else:
            q_states = torch.zeros_like(original_hidden_states)
            q_states.view(-1, dim)[gathered_mask_index] = self.proj_q(hidden_states)
            q_states = self._shape(q_states, bsz)  # (bsz, n_heads, seq_len, head_dim)

        # Get key, value proj
        if is_cross_attention:
            if remove_padding:
                raise NotImplementedError("Cross attention has not supported `remove_padding` yet")
            # Cross attention
            if past_key_value is not None:
                # Reuse k v
                k_states = past_key_value[0]
                v_states = past_key_value[1]
            else:
                if self._use_ft_linear_transpose_fusion:
                    k_states = self.proj_k(key_value_states)
                    v_states = self.proj_v(key_value_states)
                elif self._use_ft_fused_attn:
                    k_states = self.proj_k(key_value_states)
                    v_states = self.proj_v(key_value_states)
                else:
                    k_states = self._shape(self.proj_k(key_value_states), bsz)  # (bsz, n_heads, seq_len, head_dim)
                    v_states = self._shape(self.proj_v(key_value_states), bsz)  # (bsz, n_heads, seq_len, head_dim)
        else:
            # Self attention
            if not remove_padding:
                if self._use_ft_linear_transpose_fusion:
                    if self.use_moe:
                        states, self.l_aux, _ = self.proj_moe(hidden_states)
                        slice_line = states.shape[3] // 2
                        k_states, v_states = (
                            states[:, :, :, :slice_line].contiguous(),
                            states[:, :, :, slice_line:].contiguous(),
                        )
                    else:
                        k_states = self.proj_k(hidden_states)
                        v_states = self.proj_v(hidden_states)
                elif self._use_ft_fused_attn:
                    k_states = self.proj_k(hidden_states)
                    v_states = self.proj_v(hidden_states)
                else:
                    if shaped_projected_k is not None:
                        k_states = shaped_projected_k
                        v_states = shaped_projected_v
                    else:
                        if self.use_moe:
                            states, self.l_aux, _ = self.proj_moe(hidden_states)
                            slice_line = states.shape[2] // 2
                            k_states_tmp, v_states_tmp = (
                                states[:, :, :slice_line].contiguous(),
                                states[:, :, slice_line:].contiguous(),
                            )
                        else:
                            k_states_tmp = self.proj_k(hidden_states)
                            v_states_tmp = self.proj_v(hidden_states)
                        k_states = self._shape(k_states_tmp, bsz)
                        v_states = self._shape(v_states_tmp, bsz)
            else:
                k_states = torch.zeros_like(original_hidden_states)
                k_states.view(-1, dim)[gathered_mask_index] = self.proj_k(hidden_states)
                k_states = self._shape(k_states, bsz)  # (bsz, n_heads, seq_len, head_dim)
                v_states = torch.zeros_like(original_hidden_states)
                v_states.view(-1, dim)[gathered_mask_index] = self.proj_v(hidden_states)
                v_states = self._shape(v_states, bsz)  # (bsz, n_heads, seq_len, head_dim)
            if past_key_value is not None:
                if remove_padding:
                    raise NotImplementedError("`past_key_value` has not supported `remove_padding` yet")
                if self._use_ft_fused_attn:
                    raise NotImplementedError("`past_key_value` has not supported `use_ft_fused_attn` yet")
                # Reuse k v
                k_states = torch.cat([past_key_value[0], k_states], dim=2)
                v_states = torch.cat([past_key_value[1], v_states], dim=2)

        if self.config.pos_emb_type == "roformer":
            ro_q = torch.stack([-q_states[..., 1::2], q_states[..., ::2]], dim=-1).reshape_as(q_states)
            ro_k = torch.stack([-k_states[..., 1::2], k_states[..., ::2]], dim=-1).reshape_as(k_states)
            q_states = q_states * self.cos_pos[:, :, :tgt_len] + ro_q * self.sin_pos[:, :, :tgt_len]
            k_states = k_states * self.cos_pos[:, :, :tgt_len] + ro_k * self.sin_pos[:, :, :tgt_len]

        if self._use_ft_fused_attn:
            assert attn_bias is None, "FT FusedAttn does not support attn_bias"
            assert attn_mask is not None
            if attn_mask.dim() == 4:  # (bsz, 1, seqlen, seqlen)
                attn_mask = attn_mask.squeeze(1)  # (bsz, seqlen, seqlen), fp16
            assert attn_mask.dim() == 3
            attn_mask = attn_mask.contiguous().eq(0).to(dtype=q_states.dtype)
            attn_outputs = self.faster_attn(q_states, k_states, v_states, attn_mask)
            return MHAOutput(
                attn_outputs=attn_outputs,
                attn_probs=None,
                attn_weights=None,
                q_states=None if self.config.omit_other_attn_output else q_states,  # (bsz, n_heads, seq_len, head_dim)
                k_states=None
                if (self.config.omit_other_attn_output and not self.is_decoder)
                else k_states,  # (bsz, n_heads, seq_len, head_dim)
                v_states=None
                if (self.config.omit_other_attn_output and not self.is_decoder)
                else v_states,  # (bsz, n_heads, seq_len, head_dim)
                attn_bias=attn_bias,
            )

        src_len = k_states.size(-2)

        unite_d01 = self.config.mha_acts_unite_d01
        n_heads = self.config.n_heads
        if self._use_mp_linear:
            n_heads = n_heads // self.model_parallel_size
        proj_shape = (bsz * n_heads, -1, self.config.head_dim)

        if self.config.use_ft_mm_in_attn and (
            tgt_len == src_len
        ):  # TODO: fix when tgt_len != src_len, which means decoder cross_attention
            # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S)
            if self.config.use_ft_mm_in_attn_wo_scale:
                attn_weights = self.faster_matmul(q_states * self.score_scale, k_states, transpose_b=True)
            else:
                attn_weights = self.faster_matmul(
                    q_states,
                    k_states,
                    transpose_b=True,
                    scale=self.score_scale,
                )
            if unite_d01:
                attn_weights = attn_weights.view(bsz * n_heads, tgt_len, src_len)
        else:
            if unite_d01:
                attn_weights = torch.bmm(
                    q_states.view(proj_shape) * self.score_scale,
                    k_states.view(proj_shape).transpose(-1, -2),
                )  # (bsz * n_heads, seq_len, seq_len)
            else:
                attn_weights = torch.matmul(
                    q_states * self.score_scale, k_states.transpose(-1, -2)
                )  # (bsz, n_heads, seq_len, seq_len)

        if self.config.use_ft_softmax and (attn_mask is not None):  # TODO: fix what if attn_mask none
            raise NotImplementedError("FT not supported in titan.")
            # attn_weights_for_ft_softmax = attn_weights.view(bsz, n_heads, tgt_len, src_len) if unite_d01 else attn_weights
            # if attn_bias is not None:
            #     attn_weights_for_ft_softmax = attn_weights_for_ft_softmax + attn_bias
            # if FT_TRAIN_LIB_VER < 3:
            #     attn_mask = attn_mask.repeat(1, n_heads, 1, 1)
            #     # attn_mask = attn_mask.expand([-1, n_heads, -1, -1])  # TODO: results in `RuntimeError: mask must be contiguous`
            #     attn_probs = self.faster_softmax(
            #         attn_weights_for_ft_softmax,
            #         attn_mask,
            #         self.config.p_drop_attn if (self.training and not self.config.disable_ft_softmax_dropout) else .0,
            #     )
            # else:
            #     if attn_mask.dim() == 4:  # (bsz, 1, seqlen, seqlen)
            #         attn_mask = attn_mask.squeeze(1)  # (bsz, seqlen, seqlen), fp16
            #     assert attn_mask.dim() == 3
            #     attn_mask = attn_mask.contiguous().eq(0).to(dtype=q_states.dtype)
            #     attn_probs = self.faster_softmax(
            #         attn_weights_for_ft_softmax,
            #         attn_mask,
            #         n_heads,
            #         self.config.p_drop_attn if (self.training and not self.config.disable_ft_softmax_dropout) else .0,
            #         True,
            #     )
            # if unite_d01:
            #     attn_probs = attn_probs.view(bsz * n_heads, tgt_len, src_len)
            # attn_weights = None
        else:
            if attn_bias is not None:
                if unite_d01:
                    attn_weights = attn_weights.view(bsz, n_heads, tgt_len, src_len) + attn_bias
                    attn_weights = attn_weights.view(
                        bsz * n_heads, tgt_len, src_len
                    )  # (bsz * n_heads, seq_len, seq_len)
                else:
                    if self.training:
                        attn_weights = attn_weights + attn_bias
                    else:
                        attn_weights = hack_torch_trace(
                            attn_weights, attn_bias
                        )  # HACK: walk-around for torch.jit.trace bug
            if attn_mask is not None:
                if unite_d01:
                    attn_weights = attn_weights.view(bsz, n_heads, tgt_len, src_len) + attn_mask
                    attn_weights = attn_weights.view(
                        bsz * n_heads, tgt_len, src_len
                    )  # (bsz * n_heads, seq_len, seq_len)
                else:
                    if self.training:
                        attn_weights = attn_weights + attn_mask
                    else:
                        attn_weights = hack_torch_trace(
                            attn_weights, attn_mask
                        )  # HACK: walk-around for torch.jit.trace bug

            if self.config.use_realformer and (prev_attn_weights is not None):
                if unite_d01:
                    attn_weights = attn_weights + prev_attn_weights.view(
                        bsz * n_heads, tgt_len, src_len
                    )  # (bsz * n_heads, seq_len, seq_len)
                else:
                    attn_weights = attn_weights + prev_attn_weights

            # if self.config.clamp_inf_nan:
            #     attn_weights = rescue_inf_nan(attn_weights)

            attn_probs = F.softmax(attn_weights, dim=-1)  # (bsz * n_heads, seq_len, seq_len)
            attn_probs = self.dropout(attn_probs)  # (bsz * n_heads, seq_len, seq_len)

        if self.config.use_ft_mm_in_attn:
            # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W)
            if unite_d01:
                attn_outputs = self.faster_matmul(attn_probs.view(bsz, n_heads, tgt_len, src_len), v_states)
                attn_outputs = attn_outputs.view(proj_shape)
            else:
                attn_outputs = self.faster_matmul(attn_probs, v_states)
        else:
            # (bsz * n_heads, tgt_seq_len, src_seq_len) * (bsz * n_heads, src_seq_len, head_dim) => (bsz * n_heads, tgt_seq_len, head_dim)
            if unite_d01:
                attn_outputs = torch.bmm(attn_probs, v_states.view(proj_shape))
            else:
                attn_outputs = torch.matmul(attn_probs, v_states)
        attn_outputs = self._shape2(attn_outputs, bsz)  # (bsz, seq_len, dim)
        if remove_padding:
            attn_outputs = attn_outputs.view(-1, dim)[gathered_mask_index]
        return MHAOutput(
            attn_outputs=attn_outputs,
            attn_probs=None
            if self.config.omit_other_attn_output
            else (
                attn_probs if not unite_d01 else attn_probs.view(bsz, n_heads, tgt_len, src_len)
            ),  # (bsz, n_heads, seq_len, seq_len)
            attn_weights=None
            if (self.config.omit_other_attn_output or self.config.use_ft_softmax)
            else (
                attn_weights if not unite_d01 else attn_weights.view(bsz, n_heads, tgt_len, src_len)
            ),  # (bsz, n_heads, seq_len, seq_len)
            q_states=None if self.config.omit_other_attn_output else q_states,  # (bsz, n_heads, seq_len, head_dim)
            k_states=None
            if (self.config.omit_other_attn_output and not self.is_decoder)
            else k_states,  # (bsz, n_heads, seq_len, head_dim)
            v_states=None
            if (self.config.omit_other_attn_output and not self.is_decoder)
            else v_states,  # (bsz, n_heads, seq_len, head_dim)
            attn_bias=attn_bias,
        )


class TransformerEncoderLayer(nn.Module):
    attention_class = MultiHeadAttention

    def __init__(self, config: TransformerConfig, order: int = -1):
        super().__init__()
        if isinstance(config, dict):
            config = TransformerConfig(**config)
        self.config = config
        self.dim = self.config.dim
        self.l_aux = 0
        self.l_aux_attn = 0
        self.order = order

        self.attn = self.attention_class(config, order=order)
        self.fused_self_attn = None  # TODO: for ft inference

        if self.config.use_ft_attn_out_proj_dropout_fusion:
            assert self.config.use_ft_linear_in_attn_out
        if self.config.use_mp_linear_in_attn_out:
            assert RowParallelLinear is not None
            assert self.config.use_mp_linear_in_attn
        if self.config.use_ft_linear_in_attn_out:
            assert FTLinear is not None
            proj_cls = FTLinear
            dropout_rate = 0.0 if not self.config.use_ft_attn_out_proj_dropout_fusion else config.p_drop_hidden
            proj_kwargs = {"dropout_rate": dropout_rate}
            if self.config.use_mp_linear_in_attn_out:
                proj_cls = RowParallelLinear
                proj_kwargs["use_ft"] = True
            self.proj = proj_cls(config.dim, config.dim, **proj_kwargs)
        else:
            if self.config.use_mp_linear_in_attn_out:
                self.proj = RowParallelLinear(config.dim, config.dim)
            else:
                self.proj = nn.Linear(config.dim, config.dim)

        if self.config.use_ft_layernorm:
            raise NotImplementedError("FT is not enabled in titan.")

        if self.config.use_ft_layernorm:
            self.norm1 = FTLayerNorm(config.dim)
        else:
            self.norm1 = LayerNormTypes[config.layernorm_type](config.dim, config.layer_norm_eps)

        self.pwff = PositionWiseFeedForward(config)
        if self.config.use_moe and str(order) in self.config.use_moe_transformer_layer.split(","):
            import janus.groups
            import janus.layer

            if janus.groups.is_initialized() and janus.groups.get_ep_size() > 1:
                assert config.moe_experts % janus.groups.get_ep_size() == 0, "num_expert must divide moe_ep_size"
            self._use_moe = True
            if config.moe_warmup_stage:
                self.moe_warmup_k = list(map(int, config.moe_warmup_stage.split(",")))
                self.moe_warmup_steps = list(map(int, config.moe_warmup_steps.split(",")))
            else:
                self.moe_warmup_k = None
            self._use_moe_lego = config.use_moe_lego
            if config.use_moe_lego:
                self.pwff = janus.layer.MoE(
                    hidden_size=config.moe_dim,
                    expert=None,
                    num_experts=config.moe_experts,
                    k=config.moe_k,
                    use_lego=True,
                    lego_dropout_rate=0.0,
                    lego_activation=config.act,
                    is_dropout=config.use_ffn_output_dropout,
                    output_dropout_prob=config.p_drop_hidden,
                )
            else:
                self.pwff = janus.layer.MoE(
                    hidden_size=config.moe_dim,
                    expert=self.pwff,
                    num_experts=config.moe_experts,
                    k=config.moe_k,
                    noisy_gate_policy=config.moe_noisy_gate_policy,
                    load_balanced=config.moe_load_balanced,
                    enable_token_drop=config.moe_enable_token_drop,
                    expert_shape=config.moe_expert_shape,
                )
        else:
            self._use_moe = False

        if self.config.use_ft_layernorm:
            self.norm2 = FTLayerNorm(config.dim)
        else:
            self.norm2 = LayerNormTypes[config.layernorm_type](config.dim, config.layer_norm_eps)
        self.dropout1 = nn.Dropout(config.p_drop_hidden)
        self.dropout2 = nn.Dropout(config.p_drop_hidden2)

        if config.layernorm_fp16:
            self.norm1._simply_cast = True
            self.norm2._simply_cast = True

    def _hook_hidden_a_attn_out_b_res1(self, hidden_states, **kwargs):
        return hidden_states

    def _hook_hidden_a_ffn_b_res2(self, hidden_states, **kwargs):
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor,
        prev_attn_weights: Optional[torch.Tensor] = None,
        gathered_mask_index: Optional[torch.Tensor] = None,
        gathered_hidden_states: Optional[torch.Tensor] = None,
        attn_mask_for_remove_pad: Optional[torch.Tensor] = None,
        word_index_for_remove_pad: Optional[torch.Tensor] = None,
        q_state: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        ffn_type_ids: Optional[torch.Tensor] = None,
    ) -> TransformerEncoderLayerOutput:
        """Transformer layer (encoder)

        Args:
            hidden_states (torch.Tensor): (bsz, seq_len, dim)
            attn_mask (torch.Tensor): (bsz, 1, tgt_len, src_len)
            prev_attn_weights (torch.Tensor, optional): (realformer) previous attention_weights. (bsz, n_heads, seq_len, seq_len)

        Returns:
            hidden_states (torch.Tensor): (bsz, seq_len, dim)
            attn_probs (torch.Tensor): (bsz, n_heads, seq_len, seq_len)
            attn_weights (torch.Tensor): (bsz, n_heads, seq_len, seq_len)
            (q_states, k_states, v_states): (bsz, n_heads, seq_len, head_dim)
        """
        bsz = hidden_states.size(0)
        seq_len = hidden_states.size(1)
        seq_len_padded_even = False

        # Multi-Head Self-Attention
        original_hidden_states = hidden_states
        remove_padding = gathered_hidden_states is not None
        if remove_padding:
            assert not self.config.use_ft_remove_pad
            hidden_states = gathered_hidden_states

        if self.config.pad_seq_len_even:
            assert not remove_padding
            assert not self.config.use_ft_remove_pad
            if seq_len % 2 == 1:
                seq_len_padded_even = True
                # (bsz, seqlen, dim) -> (bsz, seqlen+1, dim)
                hidden_states = F.pad(hidden_states, (0, 0, 0, 1), "constant", 0)
                # (bsz, 1, seqlen, seqlen) -> (bsz, 1, seqlen+1, seqlen+1)
                attn_mask = F.pad(
                    attn_mask,
                    (0, 1, 0, 1),
                    "constant",
                    torch.finfo(attn_mask.dtype).min,
                )
                if attn_bias is not None:
                    attn_bias = F.pad(
                        attn_bias,
                        (0, 1, 0, 1),
                        "constant",
                        torch.finfo(attn_bias.dtype).min,
                    )

        word_idx = None
        if self.config.use_ft_remove_pad:
            raise NotImplementedError("FT not supported in titan.")
            # if attn_mask_for_remove_pad is None:
            #     attn_mask_for_remove_pad = attn_mask.squeeze()  # (bsz, 1, seq_len, seq_len) -> (bsz, seq_len, seq_len)
            #     attn_mask_for_remove_pad = attn_mask_for_remove_pad.eq(0).to(dtype=attn_mask.dtype)
            # assert attn_mask_for_remove_pad.dim() == 3
            # if word_index_for_remove_pad is not None:
            #     word_idx = word_index_for_remove_pad
            # else:
            #     word_idx = Get_valid_word_index(attn_mask_for_remove_pad)

        residual = hidden_states if q_state is None else q_state  # TODO: ensure
        if self.config.use_ft_remove_pad:
            raise NotImplementedError("FT not supported in titan.")
            # residual = Compress_input(residual, word_idx)

        if self.config.use_pre_layernorm:
            hidden_states = self.norm1(hidden_states)

        self_attn_output: MHAOutput = self.attn(
            hidden_states if not remove_padding else original_hidden_states,
            attn_mask=attn_mask,
            prev_attn_weights=prev_attn_weights,
            gathered_mask_index=gathered_mask_index,
            gathered_hidden_states=gathered_hidden_states,
            q_state=q_state,
            attn_bias=attn_bias,
        )
        if self.training:
            self.l_aux_attn = self.attn.l_aux
        self_attn_output_hidden = self_attn_output.attn_outputs
        if self.config.use_ft_remove_pad:
            raise NotImplementedError("FT not supported in titan.")
            # self_attn_output_hidden = Compress_input(self_attn_output_hidden, word_idx)

        hidden_states = self.proj(self_attn_output_hidden)
        if not self.config.use_ft_attn_out_proj_dropout_fusion:
            hidden_states = self.dropout1(hidden_states)

        # After out proj (including the last dropout), before the (first) residual addition
        hidden_states = self._hook_hidden_a_attn_out_b_res1(hidden_states)

        if self.config.use_deep_norm:
            residual = residual * self.config.deep_norm_enc_alpha

        if self.config.use_ft_layernorm and (not self.config.use_pre_layernorm):
            hidden_states = self.norm1(hidden_states, residual)
        else:
            hidden_states = residual + hidden_states
            if not self.config.use_pre_layernorm:
                hidden_states = self.norm1(hidden_states)

        # Position-wise Feed-Forward Networks
        residual = hidden_states
        if self.config.use_pre_layernorm:
            hidden_states = self.norm2(hidden_states)
        if self._use_moe:
            # hidden_states = self.pwff(hidden_states)
            # l_aux = self.pwff.l_aux
            import janus.groups

            if self.moe_warmup_k:
                now_step = janus.groups.get_step()
                k = -1
                for i in range(len(self.moe_warmup_steps)):
                    if now_step >= self.moe_warmup_steps[i]:
                        k = self.moe_warmup_k[i]
                self.pwff.set_topk(k)
            hidden_states, l_aux, _ = self.pwff(hidden_states)
            if self.training:
                self.l_aux = l_aux
                if not self._use_moe_lego:
                    self.z_loss = self.pwff.get_z_loss()
                else:
                    self.z_loss = torch.tensor(
                        0,
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )
        else:
            if (ffn_type_ids is None) or not getattr(self, "_use_n_ffn_types", False):  # TODO
                hidden_states = self.pwff(hidden_states)
            else:
                ffn_type_dim = ffn_type_ids.dim()
                # TODO: computation efficiency
                if ffn_type_dim == 2:  # shape of [bsz, seqlen], token-level moe
                    hidden_states_blank = torch.zeros_like(hidden_states)
                    for ffn_type_idx in range(len(self.pwff)):
                        # cur_hidden = torch.zeros_like(hidden_states)
                        # cur_ffn_type_index = ffn_type_ids == ffn_type_idx
                        # cur_ffn_type_state = hidden_states[cur_ffn_type_index]
                        # if torch.any(cur_ffn_type_index):
                        #     # cur_hidden[cur_ffn_type_index] = self.pwff[ffn_type_idx](cur_ffn_type_state.unsqueeze(0)).squeeze(0)
                        #     cur_hidden[cur_ffn_type_index] = self.pwff[ffn_type_idx](cur_ffn_type_state)
                        #     hidden_states_blank += cur_hidden
                        cur_hidden = self.pwff[ffn_type_idx](hidden_states)
                        cur_hidden[ffn_type_ids != ffn_type_idx] = 0
                        hidden_states_blank += cur_hidden
                    hidden_states = hidden_states_blank
                elif ffn_type_dim == 1:  # shape of [bsz], sequence-level moe
                    hidden_states_blank = torch.zeros_like(hidden_states)
                    for ffn_type_idx in range(len(self.pwff)):
                        cur_hidden = self.pwff[ffn_type_idx](hidden_states)
                        cur_hidden[ffn_type_ids != ffn_type_idx] = 0
                        hidden_states_blank += cur_hidden
                    hidden_states = hidden_states_blank
                elif ffn_type_dim == 0:  # shape of [], batch-level moe
                    ffn_type_idx = int(ffn_type_ids)
                    hidden_states = self.pwff[ffn_type_idx](hidden_states)
                else:
                    raise Exception(f"Unsupported dim of ffn_type_ids: {ffn_type_dim}")

        if not self.config.use_ffn_output_dropout:
            hidden_states = self.dropout2(hidden_states)

        # After ffn (including the last dropout), before the (second) residual addition
        hidden_states = self._hook_hidden_a_ffn_b_res2(hidden_states)

        if self.config.use_deep_norm:
            residual = residual * self.config.deep_norm_enc_alpha

        if self.config.use_ft_layernorm and (not self.config.use_pre_layernorm):
            raise NotImplementedError("FT not supported in titan.")
            # hidden_states = self.norm2(hidden_states, residual)
        else:
            hidden_states = residual + hidden_states
            if not self.config.use_pre_layernorm:
                hidden_states = self.norm2(hidden_states)

        if self.config.use_ft_remove_pad:
            raise NotImplementedError("FT not supported in titan.")
            # hidden_states = Restore_output(hidden_states, word_idx, bsz, seq_len)

        if self.config.clamp_inf_nan and (hidden_states.dtype == torch.float16):
            hidden_states = rescue_inf_nan(hidden_states)

        attn_bias = self_attn_output.attn_bias
        attn_probs = self_attn_output.attn_probs
        attn_weights = self_attn_output.attn_weights
        q_states = self_attn_output.q_states
        k_states = self_attn_output.k_states
        v_states = self_attn_output.v_states
        if seq_len_padded_even:
            hidden_states = hidden_states[:, :-1, :]
            if attn_bias is not None:
                attn_bias = attn_bias[:, :, :-1, :-1]
            if attn_probs is not None:
                attn_probs = attn_probs[:, :, :-1, :-1]
            if attn_weights is not None:
                attn_weights = attn_weights[:, :, :-1, :-1]
            if q_states is not None:
                q_states = q_states[..., :-1, :]
            if k_states is not None:
                k_states = k_states[..., :-1, :]
            if v_states is not None:
                v_states = v_states[..., :-1, :]
        return TransformerEncoderLayerOutput(
            hidden_states=hidden_states,
            attn_probs=attn_probs,
            attn_weights=attn_weights,
            q_states=q_states,
            k_states=k_states,
            v_states=v_states,
            attn_bias=attn_bias,
        )
