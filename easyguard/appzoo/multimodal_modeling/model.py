# -*- coding: utf-8 -*-

import json
import math
import os
from collections import OrderedDict
from types import SimpleNamespace
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from ptx.model import Model
from torch.optim.lr_scheduler import _LRScheduler

try:
    import cruise
except ImportError:
    print(
        "[ERROR] cruise is not installed! Please refer this doc: https://bytedance.feishu.cn/wiki/wikcnGP7yzZAuKpPfL6jRJKl2ag"
    )

from cruise import CruiseModule
from cruise.utilities.cloud_io import load
from cruise.utilities.hdfs_io import hexists, hopen

from easyguard.modelzoo.models.falbert.albert import ALBert
from easyguard.modelzoo.models.falbert.swin import SwinTransformer

from ...core import AutoModel
from ...modelzoo.modeling_utils import ModelBase
from ...utils.losses import LearnableNTXentLoss, LearnablePCLLoss, SCELoss


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    optimizer (Optimizer): Wrapped optimizer.
    first_cycle_steps (int): First cycle step size.
    cycle_mult(float): Cycle steps magnification. Default: -1.
    max_lr(float): First cycle's max learning rate. Default: 0.1.
    min_lr(float): Min learning rate. Default: 0.001.
    warmup_steps(int): Linear warmup step size. Default: 0.
    gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
    last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                + (self.max_lr - base_lr)
                * (
                    1
                    + math.cos(
                        math.pi * (self.step_in_cycle - self.warmup_steps) / (self.cur_cycle_steps - self.warmup_steps)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = (
                    int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
                )
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(
                        math.log(
                            (epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1),
                            self.cycle_mult,
                        )
                    )
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult**n - 1) / (self.cycle_mult - 1)
                    )
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class FashionBertv2(CruiseModule):
    def __init__(
        self,
        config_text: str = "hdfs://haruna/home/byte_ecom_govern/user/liuyuhang/pretrain/fashion_bert_v2/config_text.yaml",
        config_visual: str = "hdfs://haruna/home/byte_ecom_govern/user/liuyuhang/pretrain/fashion_bert_v2/config_visual.yaml",
        config_fusion: str = "hdfs://haruna/home/byte_ecom_govern/user/liuyuhang/pretrain/fashion_bert_v2/config_fusion.yaml",
        learning_rate: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.02,
    ):
        super(FashionBertv2, self).__init__()
        self.save_hparams()

    def setup(self, stage) -> None:
        self.model = AutoModel.from_pretrained("fashionbert-base")

        assert (
            hexists(self.hparams.config_text)
            and hexists(self.hparams.config_visual)
            and hexists(self.hparams.config_fusion)
        )
        with hopen(self.hparams.config_text) as fp:
            self.config_text = SimpleNamespace(**yaml.load(fp, yaml.Loader))
        with hopen(self.hparams.config_visual) as fp:
            self.config_visual = SimpleNamespace(**yaml.load(fp, yaml.Loader))
        with hopen(self.hparams.config_fusion) as fp:
            self.config_fusion = SimpleNamespace(**yaml.load(fp, yaml.Loader))

        """
        Initialize loss
        """
        self.calc_nce_loss_vt = LearnableNTXentLoss(
            init_tau=self.config_fusion.init_tau,
            clamp=self.config_fusion.tau_clamp,
        )  # 底层图-文CLIP损失
        self.calc_pcl_loss_ff = LearnablePCLLoss(
            init_tau=self.config_fusion.init_tau,
            clamp=self.config_fusion.tau_clamp,
            num_labels=self.config_fusion.category_logits_level1 + 1,
        )  # 上层融合-融合CLIP损失，使用一级标签
        self.category_pred_loss = SCELoss(
            alpha=1.0,
            beta=0.5,
            num_classes=self.config_fusion.category_logits_level2 + 1,
        )  # 融合表征的预测损失

        self.fuse_category = nn.Sequential(
            nn.Linear(
                self.config_fusion.hidden_size,
                self.config_fusion.hidden_size * 2,
            ),
            nn.GELU(),
            nn.Dropout(self.config_fusion.embd_pdrop),
            nn.Linear(
                self.config_fusion.hidden_size * 2,
                self.config_fusion.category_logits_level2 + 1,
            ),
        )

    def cal_pt_loss(self, **kwargs):
        for key in ["t_rep", "v_rep", "fuse_rep", "label_l1"]:
            if key in kwargs:
                kwargs[key] = self.all_gather(kwargs[key].contiguous())
                kwargs[key] = kwargs[key].flatten(0, 1)

        vt_loss = self.calc_nce_loss_vt.forward(t_emb=kwargs["t_rep"], v_emb=kwargs["v_rep"])

        assert "label_l1" in kwargs
        ff_loss = self.calc_pcl_loss_ff.forward(f_emb=kwargs["fuse_rep"], label=kwargs["label_l1"])

        assert "label" in kwargs and "fuse_cat" in kwargs
        cat_loss = self.category_pred_loss.forward(pred=kwargs["fuse_cat"], labels=kwargs["label"])

        loss = (vt_loss + ff_loss + cat_loss) / 3

        return {
            "loss": loss,
            "vt_loss": vt_loss,
            "ff_loss": ff_loss,
            "cat_loss": cat_loss,
        }

    def training_step(self, batch, idx):
        token_ids, image, label, label_l1 = (
            batch["token_ids"],
            batch["image"],
            batch["label"],
            batch["label_l1"],
        )
        rep_dict = self.model(token_ids, image)
        rep_dict.update(batch)
        rep_dict["fuse_cat"] = self.fuse_category(rep_dict["fuse_emb"][:, 0])  # [B, num_categories]

        loss_dict = self.cal_pt_loss(**rep_dict)

        return loss_dict

    def validation_step(self, batch, idx):
        return self.training_step(batch, idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=self.hparams.betas,
            eps=self.hparams.eps,
            weight_decay=self.hparams.weight_decay,
        )

        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.trainer.total_steps,
            cycle_mult=1.0,
            max_lr=self.hparams.learning_rate,
            min_lr=0,
            warmup_steps=2000,
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


class AttentionPooler(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooler, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(self, x):
        alpha = F.softmax(self.mlp(x), dim=-2)
        out = (alpha * x).sum(-2)

        return out


class AttMaxPooling(nn.Module):
    """
    使用Attention Aggregation + Max Pooling来聚合多个特征，得到一个融合特征；
    使用FC层映射到新的空间，用于对比学习；
    """

    def __init__(self, input_dim, output_dim, dropout=0.1):
        super(AttMaxPooling, self).__init__()

        self.att_pool = AttentionPooler(input_dim)
        self.max_pool = nn.AdaptiveMaxPool1d(output_size=1)

        self.fc = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x):
        """
        :param x: [B, N, d_h]
        """
        x_att = self.att_pool(x)  # [B, d_h]
        x_max = self.max_pool(x.transpose(1, 2)).squeeze()  # [B, d_h]
        x_out = self.fc(torch.cat([x_att, x_max], dim=-1))

        return x_out


class FashionProduct(CruiseModule):
    def __init__(
        self,
        config_visual: str = "./examples/fashionproduct/configs/config_visual.yaml",
        config_fusion: str = "./examples/fashionproduct/configs/config_fusion.yaml",
        learning_rate: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.02,
    ):
        super(FashionProduct, self).__init__()
        self.save_hparams()

    def setup(self, stage) -> None:
        self.model = AutoModel.from_pretrained("fashionproduct-base")

        assert hexists(self.hparams.config_visual) and hexists(self.hparams.config_fusion)
        with hopen(self.hparams.config_visual) as fp:
            self.config_visual = SimpleNamespace(**yaml.load(fp, yaml.Loader))
        with hopen(self.hparams.config_fusion) as fp:
            self.config_fusion = SimpleNamespace(**yaml.load(fp, yaml.Loader))

        """
        Pretraining mode, prepare loss functions
        """
        # (1) 1 modality-level clip loss
        init_tau = self.config_fusion.init_tau
        tau_clamp = self.config_fusion.tau_clamp
        self.modality_clip = LearnableNTXentLoss(init_tau, tau_clamp)

        # (2) 1 category-level loss
        self.category_logits = nn.Sequential(
            nn.Linear(
                self.config_fusion.hidden_size,
                self.config_fusion.hidden_size * 2,
            ),
            nn.GELU(),
            nn.Dropout(self.config_fusion.embd_pdrop),
            nn.Linear(
                self.config_fusion.hidden_size * 2,
                self.config_fusion.category_logits_level2 + 1,
            ),
        )  # 预测二级类目
        self.category_sce = SCELoss(
            alpha=1.0,
            beta=0.5,
            num_classes=self.config_fusion.category_logits_level2 + 1,
        )  # 二级类目预测损失

        # (3) 1 property prediction loss. Use following properties as targets.
        ner_tasks = self.config_fusion.ner_tasks
        ner_task_dict = self.config_fusion.ner_task_dict
        self.ner_task_dict = None
        self.ner_tasks = ner_tasks
        self.ner_heads = None
        self.ner_kl = nn.KLDivLoss(reduction="batchmean")
        if len(ner_tasks) > 0:
            assert hexists(ner_task_dict), "ner task dict {} does not exist!".format(ner_task_dict)
            with hopen(ner_task_dict, "r") as fp:
                self.ner_task_dict = json.load(fp)
            for task in ner_tasks:
                assert task in self.ner_task_dict, "task {} is not supported in ner tasks.".format(task)
            self.ner_heads = nn.ModuleList(
                [
                    AttMaxPooling(
                        self.config_fusion.hidden_size,
                        len(self.ner_task_dict[task]["label2idx"]) + 1,
                    )
                    for task in ner_tasks
                ]
            )

    def allgather(self, x):
        return self.all_gather(x.contiguous()).flatten(0, 1)

    def forward_pretrain(
        self,
        main_images,
        desc_images,
        sku_images,
        main_ocrs,
        desc_ocrs,
        sku_ocrs,
        product_name,
        other_text,
        **kwargs,
    ):
        """
        :param main_images: [B, num_main_images, 3, 244, 244]
        :param desc_images: [B, num_desc_images, 3, 244, 244]
        :param sku_images: [B, num_sku_images, 3, 244, 244]
        :param main_ocrs: [B, num_main_ocrs, max_main_len]
        :param desc_ocrs: [B, num_desc_ocrs, max_desc_len]
        :param sku_ocrs: [B, num_sku_ocrs, max_sku_len]
        :param product_name: [B, max_product_len]
        :param other_text: [B, max_other_len]
        :param fuse_mask: [B, num_segments], 1 indicates valid segment, 0 indicates mask
        """

        rep_dict = self.model.forward_fuse(
            main_images,
            desc_images,
            sku_images,
            main_ocrs,
            desc_ocrs,
            sku_ocrs,
            product_name,
            other_text,
            **kwargs,
        )

        fuse_image = rep_dict["fuse_image"]
        fuse_text = rep_dict["fuse_text"]
        fuse_image, fuse_text = self.allgather(fuse_image), self.allgather(fuse_text)
        loss_clip_modality = self.modality_clip(fuse_image, fuse_text)

        """
        3. Category-level loss.
        """
        fuse_emb = rep_dict["fuse_emb"]  # [B, 1 + 38, d_f]
        logits_cat = self.category_logits(fuse_emb[:, 0])  # [B, num_categories + 1]
        loss_sce = self.category_sce(logits_cat, kwargs["label"])

        """
        4. Property prediction loss.
        """
        # KLDiv Loss for ner
        loss_ner = 0.0
        if len(self.ner_tasks) > 0:
            for i in range(len(self.ner_tasks)):
                key = "ner_{}".format(i)
                logits = self.ner_heads[i](fuse_emb)
                label = kwargs[key]
                loss_ner = loss_ner + self.ner_kl(F.log_softmax(logits, dim=-1), label)
            loss_ner = loss_ner / len(self.ner_tasks)

        """
        5. Gather all losses.
        """
        loss = loss_clip_modality + loss_sce + loss_ner

        return {
            "loss": loss,
            "loss_clip_modality": loss_clip_modality,
            "loss_sce": loss_sce,
            "loss_ner": loss_ner,
            "logits": logits_cat,
            "label": kwargs["label"],
        }

    def training_step(self, batch, idx):
        return self.forward_pretrain(**batch)

    @torch.no_grad()
    def validation_step(self, batch, idx):
        return self.forward_pretrain(**batch)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=self.hparams.betas,
            eps=self.hparams.eps,
            weight_decay=self.hparams.weight_decay,
        )

        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.trainer.total_steps,
            cycle_mult=1.0,
            max_lr=self.hparams.learning_rate,
            min_lr=0,
            warmup_steps=2000,
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
