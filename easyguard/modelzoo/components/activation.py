"""
Activation functions, only for element-wise here.
"""

import math

import torch
import torch.nn.functional as F


def swish(x):
    return x * torch.sigmoid(x)


def mish(x):
    return x * torch.tanh(torch.nn.functional.softplus(x))


class Swish:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return swish(tensor)


class Mish:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return mish(tensor)


def _gelu_python(x):
    """Original Implementation of the gelu activation function in Google Bert repo when initially created.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    This is now written in C in torch.nn.functional
    Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


gelu = F.gelu


class GeluOld:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return _gelu_python(tensor)


class Gelu:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return gelu(tensor)


def gelu_new(x):
    """Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
    Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class GeluNew:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return gelu_new(tensor)


class SquaredReLU:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.square(torch.relu(tensor))


Activations = {
    "linear": lambda: lambda x: x,
    "relu": torch.nn.ReLU,
    "relu6": torch.nn.ReLU6,
    "elu": torch.nn.ELU,
    "prelu": torch.nn.PReLU,
    "leaky_relu": torch.nn.LeakyReLU,
    "threshold": torch.nn.Threshold,
    "hardtanh": torch.nn.Hardtanh,
    "sigmoid": torch.nn.Sigmoid,
    "tanh": torch.nn.Tanh,
    "log_sigmoid": torch.nn.LogSigmoid,
    "softplus": torch.nn.Softplus,
    "softshrink": torch.nn.Softshrink,
    "softsign": torch.nn.Softsign,
    "tanhshrink": torch.nn.Tanhshrink,
    "swish": Swish,
    "mish": Mish,
    "gelu_old": GeluOld,
    "gelu": Gelu,
    "gelu_new": GeluNew,
    "squared_relu": SquaredReLU,
}


def Activation(name: str, **kwargs):
    return Activations[name](**kwargs)
