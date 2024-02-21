# -*- coding:utf-8 -*-
# Email:    jiangxubin@bytedance.com
# Created:  2023-03-15 16:38:57
# Modified: 2023-03-15 16:38:57
"""
Reference :https://github.com/adtalos/AITM-torch/blob/master/python/model.py
"""
import torch
from torch import nn


class Tower(nn.Module):
    def __init__(self, input_dim: int, dims=[256, 128, 64], drop_prob=[0.1, 0.3, 0.3]):
        super(Tower, self).__init__()
        self.dims = dims
        self.drop_prob = drop_prob
        self.layer = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, dims[0]),
            nn.ReLU(),
            nn.Dropout(drop_prob[0]),
            nn.BatchNorm1d(dims[0]),
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Dropout(drop_prob[1]),
            nn.BatchNorm1d(dims[1]),
            nn.Linear(dims[1], dims[2]),
            nn.ReLU(),
            nn.Dropout(drop_prob[2]),
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim=32):
        super(Attention, self).__init__()
        self.dim = dim
        self.q_layer = nn.Linear(dim, dim, bias=False)
        self.k_layer = nn.Linear(dim, dim, bias=False)
        self.v_layer = nn.Linear(dim, dim, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        Q = self.q_layer(inputs)
        K = self.k_layer(inputs)
        V = self.v_layer(inputs)
        a = torch.sum(torch.mul(Q, K), -1) / torch.sqrt(torch.tensor(self.dim))
        a = self.softmax(a)
        outputs = torch.sum(torch.mul(torch.unsqueeze(a, -1), V), dim=1)
        return outputs


class AITM(nn.Module):
    def __init__(
        self,
        input_size,
        shared_out_size=256,
        tower_dims=[256, 128, 64],
        drop_prob=[0.1, 0.3, 0.3],
    ):
        super(AITM, self).__init__()

        self.input_size = input_size
        self.shared_out_size = shared_out_size
        self.bn = nn.BatchNorm1d(self.input_size)
        self.shared = nn.Linear(self.input_size, self.shared_out_size)
        self.click_tower = Tower(self.shared_out_size, tower_dims, drop_prob)
        self.conversion_tower = Tower(self.shared_out_size, tower_dims, drop_prob)
        self.attention_layer = Attention(tower_dims[-1])

        self.info_layer = nn.Sequential(
            nn.Linear(tower_dims[-1], tower_dims[-1]),
            nn.ReLU(),
            nn.Dropout(drop_prob[-1]),
        )

        self.click_layer = nn.Sequential(nn.Linear(tower_dims[-1], 1), nn.Sigmoid())
        self.conversion_layer = nn.Sequential(nn.Linear(tower_dims[-1], 1), nn.Sigmoid())

    def forward(self, x):
        x = self.bn(x)
        x = self.shared(x)

        tower_click = self.click_tower(x)

        tower_conversion = torch.unsqueeze(self.conversion_tower(x), 1)

        info = torch.unsqueeze(self.info_layer(tower_click), 1)

        ait = self.attention_layer(torch.cat([tower_conversion, info], 1))

        click = self.click_layer(tower_click)
        conversion = self.conversion_layer(ait)

        return click, conversion


class AitmMtl(nn.Module):
    def __init__(
        self,
        input_size,
        shared_out_size=256,
        tower_dims=[256, 128, 64],
        drop_prob=[0.1, 0.3, 0.3],
    ):
        super(AitmMtl, self).__init__()

        self.input_size = input_size
        self.shared_out_size = shared_out_size
        self.bn = nn.BatchNorm1d(self.input_size)
        self.shared = nn.Linear(self.input_size, self.shared_out_size)
        self.censor_tower = Tower(self.shared_out_size, tower_dims, drop_prob)
        self.reject_tower = Tower(self.shared_out_size, tower_dims, drop_prob)
        self.attention_layer = Attention(tower_dims[-1])

        self.info_layer = nn.Sequential(
            nn.Linear(tower_dims[-1], tower_dims[-1]),
            nn.ReLU(),
            nn.Dropout(drop_prob[-1]),
        )

        self.censor_layer = nn.Linear(tower_dims[-1], 1)
        self.reject_layer = nn.Linear(tower_dims[-1], 1)

    def forward(self, x):
        x = self.bn(x)
        x = self.shared(x)

        tower_censor = self.censor_tower(x)

        tower_reject = torch.unsqueeze(self.reject_tower(x), 1)

        info = torch.unsqueeze(self.info_layer(tower_censor), 1)

        ait = self.attention_layer(torch.cat([tower_reject, info], 1))

        censor = self.censor_layer(tower_censor)
        reject = self.reject_layer(ait)

        return censor, reject
