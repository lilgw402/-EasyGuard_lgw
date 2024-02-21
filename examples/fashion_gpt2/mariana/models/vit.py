#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reference Reckon Trial:
https://reckon.bytedance.net/mlx/business/mlx/studio/project/5118/version/24/train/306279
- bvc clone search/nlp/fex /opt/tiger/fex --version 1.0.0.1280 -f
- bvc clone search/nlp/fuxi /opt/tiger/fuxi --version 1.0.0.178 -f

Fex:
- Online version: 1.0.0.1280
- Description: [deepspeed-vit] add vit checkpointing
- Publish platform:SCM平台|Branch name:deepspeed-vit|Commit ID：0d4ad40.

Fuxi:
- Version: 1.0.0.178
- Description: [deepspeed-vit] add config: project_dim, visual_output_dim
- Publish platform:SCM平台|Branch name:deepspeed-vit|Commit ID：8bcb08b.

Copied from https://code.byted.org/nlp/fex/blob/0d4ad40462bb6d19638d6e86d5a8fd663a5405c4/fex/nn/backbone/vit.py
"""

from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm
from torch.utils import checkpoint

__all__ = [
    "VisualTransformer",
]


LayerNorm = nn.LayerNorm


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


from dataclasses import dataclass


@dataclass
class AttnConfig:
    hidden_size: int
    num_attention_heads: int
    hidden_dropout_prob: float

    def __init__(
        self,
        hidden_size: int,
        hidden_dropout_prob: float,
        num_attention_heads: int,
    ):
        self.hidden_size = hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.num_attention_heads = num_attention_heads


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_mlp: int,
        attn_mask: torch.Tensor = None,
        dropout=0.0,
        layernorm_eps=1.0e-5,
        use_native_attention=False,
    ):
        super().__init__()
        self.use_native_attention = use_native_attention
        if use_native_attention:
            self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        else:
            from mariana.models.albert import MultiHeadedSelfAttention

            self.attn = MultiHeadedSelfAttention(AttnConfig(d_model, dropout, n_head))
            self.proj = nn.Linear(d_model, d_model)
        self.ln_1 = LayerNorm(d_model, eps=layernorm_eps)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_mlp)),
                    ("gelu", nn.GELU()),
                    ("dropout_1", nn.Dropout(dropout)),
                    ("c_proj", nn.Linear(d_mlp, d_model)),
                    ("dropout_2", nn.Dropout(dropout)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model, eps=layernorm_eps)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        if self.use_native_attention:
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        else:
            result = self.attn(x, self.attn_mask)
            result = self.proj(result)
            return result

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        d_mlp: int,
        attn_mask: torch.Tensor = None,
        dropout=0.0,
        layernorm_eps=1.0e-5,
        use_checkpoint=False,
        use_native_attention=True,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.use_checkpoint = use_checkpoint
        kwargs = {
            "d_model": width,
            "n_head": heads,
            "d_mlp": d_mlp,
            "attn_mask": attn_mask,
            "dropout": dropout,
            "layernorm_eps": layernorm_eps,
            "use_native_attention": use_native_attention,
        }
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(**kwargs) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint_sequential(self.resblocks, self.layers, x)
        else:
            return self.resblocks(x)


class VisualTransformer(nn.Module):
    def __init__(
        self,
        patch_length: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        d_mlp: int = None,
        dropout=0.0,
        emb_dropout=0.0,
        layernorm_eps=1.0e-5,
        use_checkpoint=False,
        use_native_attention=True,
    ):
        super().__init__()
        self.patch_length = patch_length
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(patch_length + 1, width))
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.ln_pre = LayerNorm(width)
        self.use_native_attention = use_native_attention

        if not d_mlp:
            d_mlp = width * 4

        self.transformer = Transformer(
            width,
            layers,
            heads,
            d_mlp,
            dropout=dropout,
            layernorm_eps=layernorm_eps,
            use_checkpoint=use_checkpoint,
            use_native_attention=use_native_attention,
        )

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, output_last_layer=False, return_dict=False):
        output_last_layer = return_dict  # output_last_layer deprecrated
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        bsz, cur_patch_length = x.shape[:2]
        # pos emb，如果不够的用最后一个，比较hacky，先这么用着
        pos_emb = self.positional_embedding[-1].repeat(cur_patch_length, 1)
        pos_emb[: self.patch_length + 1] = self.positional_embedding[:cur_patch_length]
        x = x + pos_emb
        x = self.emb_dropout(x)
        x = self.ln_pre(x)
        if self.use_native_attention:
            x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        if self.use_native_attention:
            x = x.permute(1, 0, 2)  # LND -> NLD

        cls_out = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            cls_out = cls_out @ self.proj
        # TODO: 这样的判断可能会影响性能？看看怎么优雅的改
        if output_last_layer:
            return {"feature_map": x, "pooled_out": cls_out}
        else:
            return cls_out


def visual_transformer_B32(
    output_dim,
    dropout=0.1,
    emb_dropout=0.1,
    layernorm_eps=1.0e-5,
    patch_length=49,
):
    """
    patch length: 是vit将图片分块的数量。默认是224x224大小的图片，按32像素分块，共分成7x7=49块。
    """
    model = VisualTransformer(
        patch_length,
        32,
        768,
        12,
        12,
        output_dim,
        dropout=dropout,
        emb_dropout=emb_dropout,
        layernorm_eps=layernorm_eps,
    )
    return model


def visual_transformer_B16(
    output_dim,
    dropout=0.1,
    emb_dropout=0.1,
    layernorm_eps=1.0e-5,
    patch_length=196,
):
    """
    patch length: 是vit将图片分块的数量。默认是224x224大小的图片，按16像素分块，共分成14x14=196块。
    """
    model = VisualTransformer(
        patch_length,
        16,
        768,
        12,
        12,
        output_dim,
        dropout=dropout,
        emb_dropout=emb_dropout,
        layernorm_eps=layernorm_eps,
    )
    return model
