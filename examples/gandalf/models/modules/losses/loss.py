# -*- coding:utf-8 -*-
# Email:    jiangxubin@bytedance.com
# Created:  2023-02-27 20:20:01
# Modified: 2023-02-27 20:20:01
import torch.nn as nn


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def __init__(
        self,
        weight=None,
        pos_weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
    ):
        super().__init__(weight, size_average, reduce, reduction, pos_weight)
