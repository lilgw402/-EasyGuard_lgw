""" Basic SE block and its varieties.
      - [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
      - [Gather-Excite: Exploiting Feature Context in
        Convolutional Neural Networks](https://arxiv.org/abs/1810.12348)

"""

from torch import nn

__all__ = ["SEBlock", "SEBNBlock", "GEPFBlock", "GEPBlock", "GEPPBlock"]


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.squeeze(x)
        y = self.excite(y)
        return x * y


class SEBNBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBNBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.BatchNorm2d(channel // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.squeeze(x)
        y = self.excite(y)
        return x * y


class GEPFBlock(nn.Module):
    def __init__(self):
        super(GEPFBlock, self).__init__()
        self.gather = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sigmoid()

    def forward(self, x):
        y = self.gather(x)
        y = self.excite(y)
        return x * y


class GEPBlock(nn.Module):
    def __init__(self, channel, feat_side):
        super(GEPBlock, self).__init__()
        self.gather = nn.Linear(feat_side * feat_side, 1, bias=False)
        self.bn = nn.BatchNorm2d(channel)
        self.excite = nn.Sigmoid()

    def forward(self, x):
        n, c, h, w = x.size()
        y = self.gather(x.view(n * c, h * w)).view(n, c, 1, 1)
        y = self.bn(y)
        y = self.excite(y)
        return x * y


class GEPPBlock(nn.Module):
    def __init__(self, channel, feat_side, reduction=16):
        super(GEPPBlock, self).__init__()
        self.gather = nn.Linear(feat_side * feat_side, 1, bias=False)
        self.bn = nn.BatchNorm2d(channel)
        self.excite = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        n, c, h, w = x.size()
        y = self.gather(x.view(n * c, h * w)).view(n, c, 1, 1)
        y = self.bn(y)
        y = self.excite(y)
        return x * y
