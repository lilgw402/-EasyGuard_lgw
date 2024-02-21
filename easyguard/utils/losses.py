# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy(
    input,
    target,
    weight=None,
    size_average=None,
    ignore_index=-100,
    reduce=None,
    reduction="mean",
):
    """Cross Entropy loss.
    Args:
        input: input tensor
        target: prediction tensor
        weight: weighted cross-entropy loss (sample level weights)
        size_average: size average
        ignore_index: ignore index
        reduction: default 'mean' reduction
    """
    return F.cross_entropy(input, target, weight, size_average, ignore_index, reduce, reduction)


class LearnableNTXentLoss(torch.nn.Module):
    def __init__(self, init_tau=0.07, clamp=4.6051):
        super().__init__()
        self.tau = torch.nn.Parameter(torch.tensor([np.log(1.0 / init_tau)], dtype=torch.float32))
        self.calc_ce = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.clamp = clamp  # 4.6051 等价于CLAMP 100, 初始值是2.6593，

    def forward(self, v_emb=None, t_emb=None, logits=None):
        """
        v_emb: batch 对比loss的一边
        t_emb: batch 对比loss的另一边
        logits: 需要计算对比loss的矩阵，default: None
        """
        self.tau.data = torch.clamp(self.tau.data, 0, self.clamp)
        if logits is None:
            bsz = v_emb.shape[0]
            v_emb = F.normalize(v_emb, dim=1)
            t_emb = F.normalize(t_emb, dim=1)
            logits = torch.mm(v_emb, t_emb.t()) * self.tau.exp()  # [bsz, bsz]
        else:
            bsz = logits.shape[0]
            logits = logits * self.tau.exp()
        labels = torch.arange(bsz, device=logits.device)  # bsz

        loss_v = self.calc_ce(logits, labels)
        loss_t = self.calc_ce(logits.t(), labels)
        loss = (loss_v + loss_t) / 2
        return loss


class LearnablePCLLoss(LearnableNTXentLoss):
    def __init__(self, init_tau=0.07, clamp=4.6051, num_labels=110):
        super(LearnablePCLLoss, self).__init__(init_tau, clamp)

        self.num_labels = num_labels

    def forward(self, f_emb=None, label=None):
        """
        f_emb: [B, d_f]
        label: [B, ]
        """

        """
        1. 根据label汇聚相同label的表征；
        """
        proto_emb = torch.index_add(
            input=torch.zeros(
                (self.num_labels, f_emb.shape[-1]),
                dtype=f_emb.dtype,
                device=f_emb.device,
            ),
            dim=0,
            index=label,
            source=f_emb,
        )  # [num_labels, d_f]
        proto_cum = torch.index_add(
            input=torch.zeros((self.num_labels, 1), dtype=f_emb.dtype, device=f_emb.device),
            dim=0,
            index=label,
            source=torch.ones((f_emb.shape[0], 1), dtype=f_emb.dtype, device=f_emb.device),
        )  # [num_labels, 1]

        proto_emb = (proto_emb / (proto_cum + 1e-6)).masked_fill(proto_cum < 0.5, 0.0)  # [num_labels, d_f]

        """
        2. 计算logits；
        """
        f_emb = F.normalize(f_emb, dim=1, eps=1e-6)
        proto_emb = F.normalize(proto_emb, dim=1, eps=1e-6)
        logits = torch.mm(f_emb, proto_emb.t()) * self.tau.exp()  # [B, num_labels]

        """
        3. 计算交叉熵损失；
        """
        loss_pcl = self.calc_ce(logits, label)

        return loss_pcl


class SCELoss(nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).to(pred.dtype)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = -1 * torch.sum(pred * torch.log(label_one_hot), dim=1)

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()

        return loss


def multilabel_cross_entropy_loss(y_pred, y_true, reduction="mean"):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], axis=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], axis=-1)
    neg_loss = torch.logsumexp(y_pred_neg, axis=-1)
    pos_loss = torch.logsumexp(y_pred_pos, axis=-1)
    loss = neg_loss + pos_loss

    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()

    return loss
