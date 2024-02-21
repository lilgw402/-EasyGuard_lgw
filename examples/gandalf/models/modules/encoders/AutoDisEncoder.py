# -*- coding:utf-8 -*-
# Email:    jiangxubin@bytedance.com
# Created:  2023-02-09 16:37:59
# Modified: 2023-02-09 16:37:59
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, feature_num, input_dim, dropout, alpha=1):
        super(ResBlock, self).__init__()
        self.linear_w = nn.Parameter(torch.randn(feature_num, input_dim, input_dim))
        self.linear_b = nn.Parameter(torch.randn(feature_num, 1, input_dim))
        nn.init.kaiming_uniform_(self.linear_w, mode="fan_in", nonlinearity="leaky_relu")
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.alpha = alpha

    def forward(self, x):
        # print('Resblok input', x.shape)
        h = torch.matmul(x, self.linear_w) + self.linear_b
        h = h + self.alpha * x
        h = self.leaky_relu(h)
        h = self.dropout(h)
        # print('Resblok output', x.shape)
        return h


class AutoDisBucketEncoder(nn.Module):
    def __init__(
        self,
        feature_num,  # 必填参数，最终网络输入的特征数量
        bucket_num=8,
        bucket_dim=128,
        layer_conf=[64, 64, 64],
        alpha=1,
        output_size=128,
        use_fc=False,
        dropout=0,
        add_block=True,
    ):
        super(AutoDisBucketEncoder, self).__init__()
        self.LeakyReLU = nn.LeakyReLU(inplace=True)
        self.Dropout = nn.Dropout(p=dropout)
        self.linear1_w = nn.Parameter(torch.randn(feature_num, 3, layer_conf[0]))
        self.linear1_b = nn.Parameter(torch.randn(feature_num, 1, layer_conf[0]))
        self.layer = (
            nn.Sequential(*[ResBlock(feature_num, layer_len, dropout=0, alpha=alpha) for layer_len in layer_conf])
            if add_block
            else nn.Identity()
        )
        self.linear2_w = nn.Parameter(torch.randn(feature_num, layer_conf[-1], bucket_num))
        self.linear2_b = nn.Parameter(torch.randn(feature_num, 1, bucket_num))

        self.emb = nn.Parameter(torch.randn(feature_num, bucket_num, bucket_dim))
        self._tau_module = nn.Parameter(torch.ones([feature_num, 1, bucket_num]))
        self.Softmax = nn.Softmax(dim=-1)
        self.fc = nn.Linear(feature_num * bucket_dim, output_size, bias=True) if use_fc else nn.Identity()
        self.output_size = output_size if use_fc else feature_num * bucket_dim

        nn.init.kaiming_uniform_(self.linear1_w, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_uniform_(self.linear2_w, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        #  b feature_num 1 3
        x = x.unsqueeze(-2)
        #  b feature_num layer_conf[0]
        x = torch.matmul(x, self.linear1_w) + self.linear1_b
        x = self.LeakyReLU(x)
        x = self.Dropout(x)
        # b feature_num 1 layer_conf[-1]
        x = self.layer(x)
        #  b feature_num 1 bucket_num
        x = torch.matmul(x, self.linear2_w) + self.linear2_b
        x = self.LeakyReLU(x)
        # b feature_num bucket_num
        x = (x * self._tau_module).squeeze(-2)
        x = self.Softmax(x)
        # b feature_num bucket_num bucket_dim
        x = x.unsqueeze(-1) * self.emb
        # b feature_num bucket_dim
        x = torch.sum(x, dim=-2)
        # b feature_num*bucket_dim
        # x = torch.flatten(x, start_dim=1)
        x = torch.flatten(
            x, start_dim=-2
        )  # TODO:Use -2 instead of 1(or 2 for sequence input) will result in the same code
        # b output_size if use_fc else feature_num*bucket_num
        x = self.fc(x)
        return x
