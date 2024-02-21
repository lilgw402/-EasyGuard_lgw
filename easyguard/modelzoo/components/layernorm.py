import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    An implementation of `Layer Normalization
    <https://www.semanticscholar.org/paper/Layer-Normalization-Ba-Kiros/97fb4e3d45bb098e27e0071448b6152217bd35a5>`_ .

    Layer Normalization stabilises the training of deep neural networks by
    normalising the outputs of neurons from a particular layer. It computes:

    output = (gamma * (tensor - mean) / (std + eps)) + beta

    Args:
        dimension : ``int``, required.
            The dimension of the layer output to normalize.

    Returns:
        The normalized layer output.
    """

    def __init__(self, dimension: int, eps: float = 1e-12, **kwargs) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dimension))
        self.bias = torch.nn.Parameter(torch.zeros(dimension))
        self.eps: float = eps

    def forward(self, tensor: torch.Tensor):  # pylint: disable=arguments-differ
        # new_tensor = tensor.type_as(self.weight)
        new_tensor = tensor.to(dtype=self.weight.dtype)

        mean = new_tensor.mean(-1, keepdim=True)
        std = new_tensor.std(-1, unbiased=False, keepdim=True)
        res = self.weight * (new_tensor - mean) / (std + self.eps) + self.bias

        # res = res.type_as(tensor)
        res = res.to(dtype=tensor.dtype)
        return res


class TFLayerNorm(nn.Module):
    """LayerNorm module in the TF style (epsilon inside the square root)."""

    def __init__(self, dimension: int, eps: float = 1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.bias = nn.Parameter(torch.zeros(dimension))
        self.eps = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight * x + self.bias


class T5LayerNorm(nn.Module):
    """LayerNorm in T5 style, also named `rms_norm` officially. No bias and no subtraction of mean."""

    def __init__(self, dimension: int, eps: float = 1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.bias = None
        self.eps = eps

    def forward(self, x):
        # layer norm should always be calculated in float32
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        # convert into float16 if necessary
        if self.weight.dtype == torch.float16:
            x = x.to(torch.float16)
        return self.weight * x


LayerNormTypes = {
    "default": nn.LayerNorm,
    "v0": LayerNorm,
    "tf": TFLayerNorm,
    "t5": T5LayerNorm,
    "fused": None,
    "ft": None,
}
