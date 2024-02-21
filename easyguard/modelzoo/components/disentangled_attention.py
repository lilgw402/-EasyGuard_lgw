from dataclasses import dataclass
from functools import lru_cache  # noqa

import numpy as np  # noqa
import torch
import torch.nn as nn
import torch.nn.functional as F

# FT ops are not enabled in titan
FTDAGather = FTLinearTranspose = FTTransposeV1 = FTMatMul = FTSoftmax = None
from .transformer_encoder_decoder import Config, MHAOutput, MultiHeadAttention


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
class DAConfig(Config):
    pos_att_type: str = "c2p|p2c"
    max_relative_positions: int = 512
    position_buckets: int = -1
    p_drop_pos: float = 0.1
    use_rel_pos_cache: bool = False
    use_ft_mm_in_dattn_bias: bool = False
    use_ft_da_gather: bool = False
    use_tricky_gather: bool = False
    obey_other_attn_output: bool = False


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
        return MHAOutput(
            attn_outputs=attn_outputs,
            attn_probs=attn_probs,
            attn_weights=attn_weights,
            q_states=q_states,
            k_states=k_states,
            v_states=v_states,
            attn_bias=None,
        )
