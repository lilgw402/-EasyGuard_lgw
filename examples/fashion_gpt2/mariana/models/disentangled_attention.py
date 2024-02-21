from dataclasses import dataclass, field
from functools import lru_cache  # noqa
from typing import List, Optional

import numpy as np  # noqa
import torch
import torch.nn as nn
import torch.nn.functional as F

from .xperf_training import FTDAGather, FTLinearTranspose, FTMatMul, FTSoftmax, FTTransposeV1

# FTDAGather = FTLinearTranspose = FTTransposeV1 = FTMatMul = FTSoftmax = None


@torch.no_grad()
def make_log_bucket_position(relative_pos: torch.Tensor, bucket_size: int, max_position: int) -> torch.Tensor:
    mid = bucket_size // 2

    # sign = np.sign(relative_pos)
    # abs_pos = np.where((relative_pos < mid) & (relative_pos > -mid), mid - 1, np.abs(relative_pos))
    # log_pos = np.ceil(np.log(abs_pos / mid) / np.log((max_position - 1) / mid) * (mid - 1)) + mid
    # bucket_pos = np.where(abs_pos <= mid, relative_pos, log_pos * sign).astype(np.int)

    sign = relative_pos.sign()
    abs_pos = torch.where(
        (relative_pos < mid) & (relative_pos > -mid),
        torch.zeros_like(relative_pos).fill_(mid - 1),
        relative_pos.abs(),
    ).float()
    log_pos = (
        torch.ceil(torch.log(abs_pos / mid) / torch.log(torch.tensor((max_position - 1) / mid)) * (mid - 1)).long()
        + mid
    )
    bucket_pos = torch.where(abs_pos <= mid, relative_pos, log_pos * sign).long()

    return bucket_pos


@lru_cache()
@torch.jit.script
@torch.no_grad()
def build_relative_position(
    query_size: int,
    key_size: int,
    bucket_size: int = -1,
    max_position: int = -1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Build relative position according to the query and key

    We assume the absolute position of query :math:`P_q` is range from (0, query_size) and the absolute position of key
    :math:`P_k` is range from (0, key_size). The relative positions from query to key is :math:`R_{q \\rightarrow k} =
    P_q - P_k` .

    Args:
        query_size (int): length of query
        key_size (int): length of key

    Return:
        torch.LongTensor: tensor with shape (1, query_size, key_size)

    """

    q_ids = torch.arange(0, query_size, device=device)
    k_ids = torch.arange(0, key_size, device=device)
    rel_pos_ids = q_ids[:, None] - k_ids.repeat(q_ids.shape[0], 1)

    if bucket_size > 0 and max_position > 0:
        # rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
        mid = bucket_size // 2
        sign = rel_pos_ids.sign()
        abs_pos = torch.where(
            (rel_pos_ids < mid) & (rel_pos_ids > -mid),
            torch.zeros_like(rel_pos_ids).fill_(mid - 1),
            rel_pos_ids.abs(),
        ).float()
        log_pos = (
            torch.ceil(torch.log(abs_pos / mid) / torch.log(torch.tensor((max_position - 1) / mid)) * (mid - 1)).long()
            + mid
        )
        bucket_pos = torch.where(abs_pos <= mid, rel_pos_ids, log_pos * sign).long()
        rel_pos_ids = bucket_pos

    rel_pos_ids = rel_pos_ids[:query_size, :]
    return rel_pos_ids


@torch.jit.script
def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos):
    return c2p_pos.expand(
        [
            query_layer.size(0),
            query_layer.size(1),
            query_layer.size(2),
            relative_pos.size(-1),
        ]
    )


@torch.jit.script
def p2c_dynamic_expand(c2p_pos, query_layer, key_layer):
    return c2p_pos.expand(
        [
            query_layer.size(0),
            query_layer.size(1),
            key_layer.size(-2),
            key_layer.size(-2),
        ]
    )


@torch.jit.script
def pos_dynamic_expand(pos_index, p2c_att, key_layer):
    return pos_index.expand(p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2)))


@dataclass
class DAConfig:
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
    mha_acts_unite_d01: bool = False
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
    pos_att_type: str = "c2p|p2c"
    max_relative_positions: int = 512
    position_buckets: int = -1
    p_drop_pos: float = 0.1
    use_rel_pos_cache: bool = False
    use_ft_mm_in_dattn_bias: bool = False
    use_ft_da_gather: bool = False
    use_tricky_gather: bool = False
    obey_other_attn_output: bool = False

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


class AttentionExpertLinearTranspose(nn.Module):
    def __init__(self, dim, dim2, n_heads, head_dim):
        super().__init__()
        self.proj_k = nn.Linear(dim, dim2)
        self.proj_v = nn.Linear(dim, dim2)
        self.n_heads = n_heads
        self.head_dim = head_dim

    def _shape(self, tensor: torch.Tensor, bsz: int) -> torch.Tensor:
        """(bsz, seq_len, dim) -> (bsz, n_heads, seq_len, head_dim)"""
        return tensor.view(bsz, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

    def forward(self, x):
        bsz = x.size(0)
        k_states = self._shape(self.proj_k(x), bsz)
        v_states = self._shape(self.proj_v(x), bsz)
        return torch.concat([k_states, v_states], dim=3)


class DisentangledMHA(nn.Module):
    """
    https://arxiv.org/abs/2006.03654
    """

    def __init__(self, config: DAConfig, **kwargs):
        if isinstance(config, dict):
            config = DAConfig(**config)
        super().__init__()
        self.config = config
        self.use_moe = False
        self.l_aux = 0
        order = kwargs.get("order", -1)

        if self.config.use_moe_attn and str(order) in self.config.use_moe_transformer_layer_attn.split(","):
            import janus.layer

            self.use_moe = True
            self.proj_moe = janus.layer.MoE(
                hidden_size=config.moe_dim_attn,
                expert=AttentionExpertLinearTranspose(config.dim, config.dim, config.n_heads, config.head_dim),
                num_experts=config.moe_experts_attn,
                k=config.moe_k_attn,
                noisy_gate_policy="None",
                load_balanced=config.moe_load_balanced_attn,
                enable_token_drop=False,
                expert_shape=config.moe_attn_expert_shape,
            )
        else:
            self.proj_k = nn.Linear(config.dim, config.dim)
            self.proj_v = nn.Linear(config.dim, config.dim)
        self.proj_q = nn.Linear(config.dim, config.dim)

        self.dropout = nn.Dropout(config.p_drop_attn)
        self.score_scale = config.head_dim**-0.5

        assert not config.mha_acts_unite_d01

        self.pos_att_type = tuple(
            [x.strip() for x in config.pos_att_type.lower().split("|")] if config.pos_att_type else []
        )
        self.max_relative_positions = config.max_relative_positions
        self.position_buckets = config.position_buckets
        if self.position_buckets < 1:
            self.pos_ebd_size = self.max_relative_positions
        else:
            self.pos_ebd_size = self.position_buckets

        self.pos_dropout = nn.Dropout(config.p_drop_pos)

        self.scale_factor = 1 + len(self.pos_att_type)
        # Override
        self.score_scale = (config.head_dim * self.scale_factor) ** -0.5

    def _shape(self, tensor: torch.Tensor, bsz: int) -> torch.Tensor:
        """(bsz, seq_len, dim) -> (bsz, n_heads, seq_len, head_dim)"""
        return tensor.view(bsz, -1, self.config.n_heads, self.config.head_dim).permute(0, 2, 1, 3).contiguous()

    def disentangled_att_bias(
        self,
        query_layer: torch.Tensor,  # (bsz, n_heads, q_len, head_dim)
        key_layer: torch.Tensor,  # (bsz, n_heads, k_len, head_dim)
        relative_pos: torch.LongTensor,
        rel_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        q_len = query_layer.size(-2)
        k_len = key_layer.size(-2)
        if relative_pos is None:
            relative_pos = build_relative_position(
                q_len,
                k_len,
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
                device=query_layer.device,
            )
        if relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim() == 3:
            relative_pos = relative_pos.unsqueeze(1)
        # (b,h,q,k)
        assert relative_pos.dim() == 4, f"Relative position ids must be of dim 4 instead of {relative_pos.dim()}"

        # att_span = self.pos_ebd_size // 2  # 256
        att_span = q_len

        rel_embeddings = rel_embeddings[
            self.max_relative_positions - att_span : self.max_relative_positions + att_span,
            :,
        ]

        score = None

        # content->position
        if "c2p" in self.pos_att_type:
            if self.use_moe:
                states, _, _ = self.proj_moe(rel_embeddings, add_step=False)
                slice_line = states.shape[3] // 2
                pos_key_layer = states[:, :, :, :slice_line].contiguous()
            else:
                pos_key_layer = self.proj_k(rel_embeddings)  # (att_span*2, dim)
            pos_key_layer = pos_key_layer.view(
                -1, self.config.n_heads, self.config.head_dim
            )  # (att_span*2, n_heads, head_dim)
            c2p_att = torch.matmul(query_layer, pos_key_layer.permute(1, 2, 0))  # (bsz, n_heads, q_len, att_span*2)

            if not self.config.use_tricky_gather:
                c2p_pos = torch.clamp(relative_pos + att_span - 1, 0, att_span * 2 - 1)
                c2p_pos = c2p_pos.expand([c2p_att.size(0), c2p_att.size(1), -1, -1])
                c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_pos)
            else:
                c2p_att = c2p_att.view(c2p_att.size(0), c2p_att.size(1), -1)
                c2p_att = F.pad(c2p_att, (0, att_span), "constant", 0)
                # c2p_att = torch.cat([c2p_att, torch.zeros((1, 1, att_span), dtype=c2p_att.dtype, device=c2p_att.device).expand([c2p_att.size(0), c2p_att.size(1), -1])], dim=-1)
                # c2p_att = c2p_att.view(c2p_att.size(0), c2p_att.size(1), att_span, 2 * att_span + 1)[:, :, :, :att_span]
                # c2p_att = torch.flip(c2p_att, [-1])
                c2p_att = c2p_att.view(
                    c2p_att.size(0),
                    c2p_att.size(1),
                    att_span,
                    2 * att_span + 1,
                )[
                    ...,
                    torch.arange(att_span - 1, -1, -1, device=c2p_att.device),
                ]

            score = c2p_att

        # position->content
        if "p2c" in self.pos_att_type:
            pos_query_layer = self.proj_q(rel_embeddings)  # (att_span*2, dim)
            pos_query_layer = pos_query_layer.view(
                -1, self.config.n_heads, self.config.head_dim
            )  # (att_span*2, n_heads, head_dim)
            p2c_att = torch.matmul(key_layer, pos_query_layer.permute(1, 2, 0))  # (bsz, n_heads, k_len, att_span*2)

            if not self.config.use_tricky_gather:
                if q_len != k_len:  # TODO: ensure
                    r_pos = build_relative_position(
                        k_len,
                        k_len,
                        bucket_size=self.position_buckets,
                        max_position=self.max_relative_positions,
                        device=query_layer.device,
                    )
                    r_pos = r_pos.unsqueeze(0).unsqueeze(0)
                else:
                    r_pos = relative_pos

                p2c_pos = torch.clamp(r_pos + att_span - 1, 0, att_span * 2 - 1)
                p2c_pos = p2c_pos.expand([p2c_att.size(0), p2c_att.size(1), -1, -1])
                p2c_att = torch.gather(p2c_att, dim=-1, index=p2c_pos)
            else:
                p2c_att = p2c_att.view(p2c_att.size(0), p2c_att.size(1), -1)
                p2c_att = F.pad(p2c_att, (0, att_span), "constant", 0)
                # p2c_att = torch.cat([p2c_att, torch.zeros((1, 1, att_span), dtype=p2c_att.dtype, device=p2c_att.device).expand([p2c_att.size(0), p2c_att.size(1), -1])], dim=-1)
                # p2c_att = p2c_att.view(p2c_att.size(0), p2c_att.size(1), att_span, 2 * att_span + 1)[:, :, :, :att_span]
                # p2c_att = torch.flip(p2c_att, [-1])
                p2c_att = p2c_att.view(
                    p2c_att.size(0),
                    p2c_att.size(1),
                    att_span,
                    2 * att_span + 1,
                )[
                    ...,
                    torch.arange(att_span - 1, -1, -1, device=p2c_att.device),
                ]

            if q_len != k_len:  # TODO: ensure
                pos_index = relative_pos[:, :, :, 0].unsqueeze(-1)
                p2c_att = torch.gather(
                    p2c_att,
                    dim=-2,
                    index=pos_index.expand(p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2))),
                )

            p2c_att = p2c_att.transpose(-1, -2)
            if score is None:
                score = p2c_att
            else:
                score += p2c_att

        if score is None:
            return 0.0
        return score * self.score_scale

    def _experimental_disentangled_att_bias(
        self,
        query_layer: torch.Tensor,  # (bsz, n_heads, q_len, head_dim)
        key_layer: torch.Tensor,  # (bsz, n_heads, k_len, head_dim)
        relative_pos: torch.LongTensor,
        rel_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        q_len = query_layer.size(-2)
        k_len = key_layer.size(-2)
        if relative_pos is None:
            relative_pos = build_relative_position(
                q_len,
                k_len,
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
                device=query_layer.device,
            )
        if relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim() == 3:
            relative_pos = relative_pos.unsqueeze(1)
        # (b,h,q,k)
        assert relative_pos.dim() == 4, f"Relative position ids must be of dim 4 instead of {relative_pos.dim()}"

        # att_span = self.pos_ebd_size // 2  # 256
        att_span = q_len

        rel_embeddings = rel_embeddings[
            self.max_relative_positions - att_span : self.max_relative_positions + att_span,
            :,
        ]

        score = None

        # content->position
        if "c2p" in self.pos_att_type:
            pos_key_layer = self.proj_k(rel_embeddings)  # (att_span*2, dim)
            pos_key_layer = pos_key_layer.view(
                -1, self.config.n_heads, self.config.head_dim
            )  # (att_span*2, n_heads, head_dim)
            pos_key_layer = pos_key_layer.permute(1, 2, 0)  # (n_heads, head_dim, att_span*2)
            pos_key_layer = pos_key_layer.unsqueeze(1).expand(
                -1, q_len, -1, -1
            )  # (n_heads, q_len, head_dim, att_span*2)
            c2p_pos = torch.clamp(relative_pos + att_span - 1, 0, att_span * 2 - 1)
            c2p_pos = c2p_pos.squeeze(0).unsqueeze(2)
            c2p_pos = c2p_pos.expand([pos_key_layer.size(0), -1, pos_key_layer.size(2), -1])
            pos_key_layer = torch.gather(pos_key_layer, dim=-1, index=c2p_pos)  # (n_heads, q_len, head_dim, att_span)
            # (bsz, n_heads, q_len, head_dim) * (n_heads, q_len, head_dim, att_span) -> (bsz, n_heads, q_len, att_span)
            c2p_att = torch.matmul(query_layer.unsqueeze(3), pos_key_layer)  # (bsz, n_heads, q_len, att_span)
            c2p_att = c2p_att.squeeze(3)

            score = c2p_att

        # position->content
        if "p2c" in self.pos_att_type:
            pos_query_layer = self.proj_q(rel_embeddings)  # (att_span*2, dim)
            pos_query_layer = pos_query_layer.view(
                -1, self.config.n_heads, self.config.head_dim
            )  # (att_span*2, n_heads, head_dim)
            p2c_att = torch.matmul(key_layer, pos_query_layer.permute(1, 2, 0))  # (bsz, n_heads, k_len, att_span*2)

            if q_len != k_len:  # TODO: ensure
                r_pos = build_relative_position(
                    k_len,
                    k_len,
                    bucket_size=self.position_buckets,
                    max_position=self.max_relative_positions,
                    device=query_layer.device,
                )
                r_pos = r_pos.unsqueeze(0).unsqueeze(0)
            else:
                r_pos = relative_pos

            p2c_pos = torch.clamp(r_pos + att_span - 1, 0, att_span * 2 - 1)
            p2c_pos = p2c_pos.expand([p2c_att.size(0), p2c_att.size(1), -1, -1])
            p2c_att = torch.gather(p2c_att, dim=-1, index=p2c_pos)

            if q_len != k_len:  # TODO: ensure
                pos_index = relative_pos[:, :, :, 0].unsqueeze(-1)
                p2c_att = torch.gather(
                    p2c_att,
                    dim=-2,
                    index=pos_index.expand(p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2))),
                )

            p2c_att = p2c_att.transpose(-1, -2)
            if score is None:
                score = p2c_att
            else:
                score += p2c_att

        if score is None:
            return 0.0
        return score * self.score_scale

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor,
        relative_pos: torch.Tensor,
        relative_pos_embed: torch.Tensor,
        q_state=None,
    ) -> torch.Tensor:
        bsz = hidden_states.size(0)

        q_states = self._shape(
            self.proj_q(hidden_states if q_state is None else q_state), bsz
        )  # (bsz, n_heads, seq_len, head_dim)
        if self.use_moe:
            states, self.l_aux, _ = self.proj_moe(hidden_states)
            slice_line = states.shape[3] // 2
            k_states, v_states = (
                states[:, :, :, :slice_line].contiguous(),
                states[:, :, :, slice_line:].contiguous(),
            )
        else:
            k_states = self._shape(self.proj_k(hidden_states), bsz)
            v_states = self._shape(self.proj_v(hidden_states), bsz)

        attn_weights = torch.matmul(
            q_states * self.score_scale, k_states.permute(0, 1, 3, 2)
        )  # (bsz, n_heads, seq_len, seq_len)

        # The only diff from original MHA
        attn_weights = attn_weights + self.disentangled_att_bias(
            q_states,
            k_states,
            relative_pos,
            self.pos_dropout(relative_pos_embed),
        )

        attn_weights = attn_weights + attn_mask

        attn_probs = F.softmax(attn_weights, dim=-1)  # (bsz, n_heads, seq_len, seq_len)
        attn_probs = self.dropout(attn_probs)  # (bsz, n_heads, seq_len, seq_len)
        attn_outputs = torch.matmul(attn_probs, v_states)  # (bsz, n_heads, seq_len, head_dim)

        attn_outputs = attn_outputs.permute(0, 2, 1, 3).reshape(bsz, -1, self.config.dim)  # (bsz, seq_len, dim)
        if not self.config.obey_other_attn_output:
            return attn_outputs
        return dict(
            attn_outputs=attn_outputs,
            attn_probs=attn_probs,
            attn_weights=attn_weights,
            q_states=q_states,
            k_states=k_states,
            v_states=v_states,
            attn_bias=None,
        )


class AttentionExpertFTLinearTranspose(nn.Module):
    def __init__(self, dim, dim2, n_heads):
        super().__init__()
        self.proj_k = FTLinearTranspose(dim, dim2, n_heads, transpose_type="2013")
        self.proj_v = FTLinearTranspose(dim, dim2, n_heads, transpose_type="2013")

    def forward(self, x):
        k_states = self.proj_k(x)
        v_states = self.proj_v(x)
        return torch.concat([k_states, v_states], dim=3)


class FTDisentangledMHA(nn.Module):
    def __init__(self, config: DAConfig, **kwargs):
        if isinstance(config, dict):
            config = DAConfig(**config)
        super().__init__()
        self.config = config

        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_size = int(self.dim / self.n_heads)

        self.use_moe = False
        self.l_aux = 0
        order = kwargs.get("order", -1)
        if self.config.use_moe_attn and str(order) in self.config.use_moe_transformer_layer_attn.split(","):
            import janus.layer

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
            self.proj_k = FTLinearTranspose(self.dim, self.dim, self.n_heads, transpose_type="2013")
            self.proj_v = FTLinearTranspose(self.dim, self.dim, self.n_heads, transpose_type="2013")
        self.proj_q = FTLinearTranspose(self.dim, self.dim, self.n_heads, transpose_type="2013")

        # assert config.pos_att_type == 'c2p|p2c', 'Faster DA does only support `c2p|p2c` as `pos_att_type`'
        # self.pos_att_type = ['c2p', 'p2c']
        self.pos_att_type = tuple(
            [x.strip() for x in config.pos_att_type.lower().split("|")] if config.pos_att_type else []
        )
        self.is_deberta = len(self.pos_att_type) > 0
        self.score_scale = (self.head_size * (1 + len(self.pos_att_type))) ** -0.5

        self._use_ft_gather = config.use_ft_da_gather
        if self._use_ft_gather:
            self.faster_gather = FTDAGather(self.score_scale)
        else:
            self.faster_gather = None

        assert config.position_buckets < 1, "Faster DA does not support `position_buckets` > 0"
        self.max_relative_positions = config.max_relative_positions

        self.p_drop_attn = config.p_drop_attn
        self.pos_dropout = nn.Dropout(config.p_drop_pos)

        self.ft_trans_0213 = FTTransposeV1("0213")
        self.ft_mm = FTMatMul()
        self.ft_trans_1203 = FTTransposeV1("1203")
        self.ft_softmax = FTSoftmax()

        self._rel_pos_cache = dict()
        self._use_rel_pos_cache = config.use_rel_pos_cache

        self._pos_att_case = 0
        if "c2p" in self.pos_att_type:
            self._pos_att_case += 2**0
        if "p2c" in self.pos_att_type:
            self._pos_att_case += 2**1
        if self._use_ft_gather:
            assert self._pos_att_case % 4 == 3

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        """
        new_x_shape = x.size()[:-1] + (self.n_heads, self.head_size)
        x = x.view(*new_x_shape)
        return self.ft_trans_0213(x)

    @torch.no_grad()
    def cache_rel_pos(
        self,
        att_span: int,
        q_len: int,
        k_len: int,
        bsz: int,
        relative_pos: torch.Tensor,
    ) -> torch.Tensor:
        use_rel_pos_cache = self._use_rel_pos_cache or self.training
        if use_rel_pos_cache:
            cache_key = (q_len, k_len, bsz, att_span)
            if cache_key in self._rel_pos_cache:
                return self._rel_pos_cache[cache_key]
        rel_pos = (
            torch.clamp(relative_pos + att_span - 1, 0, att_span * 2 - 1).unsqueeze(0).unsqueeze(0)
        )  # (1, 1, q_len, k_len)
        rel_pos = rel_pos.expand([bsz, self.n_heads, q_len, -1])
        if use_rel_pos_cache:
            self._rel_pos_cache[cache_key] = rel_pos
        return rel_pos

    def disentangled_att_bias(
        self,
        original_scores: torch.Tensor,  # (bsz, n_heads, q_len, q_len)
        query_layer: torch.Tensor,  # (bsz, n_heads, q_len, head_dim)
        key_layer: torch.Tensor,  # (bsz, n_heads, k_len, head_dim)
        relative_pos: torch.LongTensor,  # (q_len, k_len)
        rel_embeddings: torch.Tensor,  # (2 * max_relative_positions, dim)
    ) -> torch.Tensor:
        q_len = query_layer.size(-2)
        k_len = key_layer.size(-2)
        att_span = q_len

        rel_embeddings = rel_embeddings[
            self.max_relative_positions - att_span : self.max_relative_positions + att_span,
            :,
        ]
        rel_embeddings = rel_embeddings.unsqueeze(0)

        bsz = query_layer.size(1)  # Actually (n_heads, bsz, seq_len, head_dim)

        # c2p
        if self._pos_att_case % 2 == 1:
            if self.use_moe:
                states, _, _ = self.proj_moe(rel_embeddings, add_step=False)
                slice_line = states.shape[3] // 2
                pos_k_layer = states[:, :, :, :slice_line].contiguous()
            else:
                pos_k_layer = self.proj_k(rel_embeddings)
            pos_k_layer = pos_k_layer.view(self.n_heads, pos_k_layer.size(-2), -1)
            query_layer = query_layer.view(self.n_heads, -1, query_layer.size(-1))
            c2p_att = self.ft_mm(
                query_layer,
                pos_k_layer.to(query_layer.dtype),
                transpose_b=True,
            )
            c2p_att = c2p_att.view(self.n_heads, bsz, q_len, c2p_att.size(-1))

        # p2c
        if self._pos_att_case % 4 >= 2:
            pos_q_layer = self.proj_q(rel_embeddings)

            pos_q_layer = pos_q_layer.view(self.n_heads, pos_q_layer.size(-2), -1)
            key_layer = key_layer.view(self.n_heads, -1, query_layer.size(-1))
            p2c_att = self.ft_mm(key_layer, pos_q_layer.to(query_layer.dtype), transpose_b=True)
            p2c_att = p2c_att.view(self.n_heads, bsz, k_len, p2c_att.size(-1))

        if self._pos_att_case == 0:
            score = original_scores
        elif self._use_ft_gather:
            score = self.faster_gather(c2p_att, p2c_att, original_scores)
        else:
            rel_pos = self.cache_rel_pos(att_span, q_len, k_len, bsz, relative_pos)
            if self._pos_att_case == 1:
                c2p_att = torch.gather(c2p_att.transpose(0, 1), dim=-1, index=rel_pos)
                score = c2p_att * self.score_scale
            elif self._pos_att_case == 2:
                p2c_att = torch.gather(p2c_att.transpose(0, 1), dim=-1, index=rel_pos)
                p2c_att = p2c_att.transpose(-1, -2)
                score = p2c_att * self.score_scale
            else:
                c2p_att = torch.gather(c2p_att.transpose(0, 1), dim=-1, index=rel_pos)
                p2c_att = torch.gather(p2c_att.transpose(0, 1), dim=-1, index=rel_pos)
                p2c_att = p2c_att.transpose(-1, -2)
                score = (c2p_att + p2c_att) * self.score_scale
            score = original_scores + score.transpose(0, 1)
        return score

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor = None,
        relative_pos: torch.Tensor = None,
        relative_pos_embed: torch.Tensor = None,
        q_state=None,
    ) -> torch.Tensor:
        tq = self.proj_q(hidden_states if q_state is None else q_state)
        if self.use_moe:
            states, self.l_aux, _ = self.proj_moe(hidden_states)
            slice_line = states.shape[3] // 2
            tk, tv = (
                states[:, :, :, :slice_line].contiguous(),
                states[:, :, :, slice_line:].contiguous(),
            )
        else:
            tk = self.proj_k(hidden_states)
            tv = self.proj_v(hidden_states)

        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = self.ft_mm(tq, tk, transpose_b=True, scale=self.score_scale)

        if self.is_deberta:
            scores = self.disentangled_att_bias(
                scores,
                tq,
                tk,
                relative_pos,
                self.pos_dropout(relative_pos_embed),
            )

        attn_mask = attn_mask.transpose(0, 1)

        if attn_mask.dim() == 4:  # (1, bsz, seqlen, seqlen)
            attn_mask = attn_mask.squeeze(0)  # (bsz, seqlen, seqlen), fp16
        assert attn_mask.dim() == 3
        attn_mask = attn_mask.contiguous().eq(0).to(dtype=hidden_states.dtype)
        probs = self.ft_softmax(scores, attn_mask, self.n_heads, self.p_drop_attn, False)

        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W)
        h = self.ft_mm(probs, tv)

        # (B, H, S, W) -trans-> (B, S, H, W) -merge-> (B, S, D)
        h = self.ft_trans_1203(h)

        new_context_layer_shape = h.size()[:-2] + (self.dim,)
        h = h.view(*new_context_layer_shape)
        if not self.config.obey_other_attn_output:
            return h
        return dict(
            attn_outputs=h,
            attn_probs=probs,
            attn_weights=scores,  # TODO: add mask?
            q_states=tq,
            k_states=tk,
            v_states=tv,
            attn_bias=None,
        )
