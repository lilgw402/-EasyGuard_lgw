import math

import torch
import torch.nn as nn


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer_norm_eps = 1e-6
        linear = nn.Linear(in_features, out_features)
        nn.init.normal_(linear.weight, std=math.sqrt(2.0 / in_features))
        nn.init.zeros_(linear.bias)
        self.model = nn.Sequential(linear, nn.GELU(), LayerNorm(out_features, eps=self.layer_norm_eps))

    def forward(self, x):
        return self.model(x)


class Prediction(nn.Module):
    """
    for concat and predict
    """

    def __init__(self, inp_features, out_features, hidden_size=768):
        super().__init__()
        self.hidden_size = hidden_size
        self.dense = nn.Sequential(
            Linear(inp_features, self.hidden_size),
            nn.Linear(self.hidden_size, out_features),
        )

    def forward(self, a, b=None):
        if b:
            return self.dense(torch.cat([a, b], dim=-1))
        else:
            return self.dense(a)
