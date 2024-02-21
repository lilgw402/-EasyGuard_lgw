# -*- coding: utf-8 -*-
"""
Video CLIP Model
"""
import math
import os.path
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

try:
    import cruise
except ImportError:
    print(
        "[ERROR] cruise is not installed! Please refer this doc: https://bytedance.feishu.cn/wiki/wikcnGP7yzZAuKpPfL6jRJKl2ag"
    )

from cruise import CruiseModule
from cruise.utilities.cloud_io import load
from cruise.utilities.distributed import DIST_ENV
from cruise.utilities.hdfs_io import hexists, hopen
from sklearn.metrics import roc_auc_score

from easyguard import AutoModel
from easyguard.appzoo.authentic_modeling.utils import CosineAnnealingWarmupRestarts, accuracy

from ...utils.losses import LearnableNTXentLoss, LearnablePCLLoss, SCELoss, cross_entropy
from .optimization import *
from .optimization import AdamW


def p_fix_r(output, labels, fix_r):
    output_sort = output[(-output).argsort()]
    labels_sort = labels[(-output).argsort()]
    num_pos = np.sum(labels == 1)
    recall_sort = np.cumsum(labels_sort) / float(num_pos)
    index = np.abs(recall_sort - fix_r).argmin()
    thr = output_sort[index]
    precision = np.sum(((output >= thr) == labels) * labels) / np.sum(output >= thr)
    return precision, recall_sort[index], thr


class HighQualityLiveVideoCLIP(CruiseModule):
    def __init__(
        self,
        config_backbone,
        config_fusion,
        config_optim,
        load_pretrained: str = None,
    ):
        super(HighQualityLiveVideoCLIP, self).__init__()
        self.save_hparams()

    def setup(self, stage) -> None:
        """
        Load yaml file as config class
        """
        with open(self.hparams.config_backbone) as fp:
            self.config_backbone = SimpleNamespace(**yaml.load(fp, yaml.Loader))
        with open(self.hparams.config_fusion) as fp:
            self.config_fusion = SimpleNamespace(**yaml.load(fp, yaml.Loader))
        with open(self.hparams.config_optim) as fp:
            self.config_optim = SimpleNamespace(**yaml.load(fp, yaml.Loader))
        """
        Initialize modules
        """
        # self.falbert = FrameALBert(self.config_backbone)
        self.falbert = AutoModel.from_pretrained(
            "/mnt/bn/ecom-govern-maxiangqian/dongjunwei/EasyGuard/easyguard/modelzoo/models/falbert_new/config"
        )
        if self.config_fusion.name == "two_stream":
            self.t_projector, self.v_projector = self.init_projector(
                input_size=self.config_fusion.input_dim,
                output_size=self.config_fusion.out_dim,
            )
        """
        Initialize output layer
        """
        if self.config_fusion.name == "two_stream":
            fuse_emb_size = self.config_fusion.out_dim * 2
        else:
            fuse_emb_size = 768
        self.classifier = torch.nn.Linear(fuse_emb_size, self.config_fusion.class_num)
        self.softmax = nn.Softmax(dim=1)
        """
        Initialize some fixed parameters.
        """
        self.init_weights()
        self.freeze_params(self.config_backbone.freeze_prefix)

    def init_weights(self):
        def init_weight_module(module):
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                torch.nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, (torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.apply(init_weight_module)

        if self.hparams.load_pretrained:
            state_dict_ori = self.state_dict()
            # load weights of pretrained model
            state_dict_new = OrderedDict()
            pretrained_weights = load(self.hparams.load_pretrained, map_location="cpu")
            if "state_dict" in pretrained_weights:
                pretrained_weights = pretrained_weights["state_dict"]
            for key, value in pretrained_weights.items():
                if key in state_dict_ori and state_dict_ori[key].shape == value.shape:
                    state_dict_new[key] = value
            missing_keys, unexpected_keys = self.load_state_dict(state_dict_new, strict=False)
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)
        else:
            state_dict_ori = self.falbert.state_dict()
            backbone = load(
                "hdfs://haruna/home/byte_search_nlp_lq/multimodal/modelhub/videoclip_swin_dy_20211206/model.th",
                map_location="cpu",
            )
            state_dict_new = OrderedDict()
            for key, value in backbone.items():
                if key.startswith("falbert"):
                    trimmed_key = key[len("falbert.") :]
                else:
                    trimmed_key = key
                if trimmed_key in state_dict_ori and state_dict_ori[trimmed_key].shape == backbone[key].shape:
                    state_dict_new[trimmed_key] = value
            missing_keys, unexpected_keys = self.falbert.load_state_dict(state_dict_new, strict=False)
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    def freeze_params(self, freeze_prefix):
        for name, param in self.named_parameters():
            for prefix in freeze_prefix:
                if name.startswith(prefix):
                    param.requires_grad = False

    def forward(
        self,
        input_ids,
        input_segment_ids,
        input_mask,
        frames=None,
        frames_mask=None,
        visual_embeds=None,
    ):
        if self.config_fusion.name == "two_stream":
            t_emb = self.encode_text(
                input_ids=input_ids,
                input_segment_ids=input_segment_ids,
                input_mask=input_mask,
            )["pooled_output"]
            t_emb = self.t_projector(t_emb)
            v_emb = self.encode_image(images=frames, images_mask=frames_mask)["pooled_output"]
            v_emb = self.v_projector(v_emb)
            cls_emb = torch.cat((t_emb, v_emb), dim=-1)
            logits = self.classifier(cls_emb)
            return {"fuse_rep": cls_emb, "out_lvl1": logits}
        else:
            mmout = self.encode_multimodal(
                input_ids=input_ids,
                input_segment_ids=input_segment_ids,
                input_mask=input_mask,
                images=frames,
                images_mask=frames_mask,
                visual_embeds=visual_embeds,
            )
            cls_emb = mmout["pooled_output"]
            logits = self.classifier(cls_emb)
            return {"fuse_rep": cls_emb, "out_lvl1": logits}

    def encode_image(
        self,
        images: torch.Tensor,
        images_mask: torch.Tensor = None,
        visual_embeds: torch.tensor = None,
    ):
        if len(images.shape) == 4:
            images = images.unsqueeze(1)
        if images_mask is None:
            if visual_embeds is None:
                images_mask = torch.ones(images.shape[0:2], device=images.device, dtype=torch.long)
            else:
                images_mask = torch.ones(
                    visual_embeds.shape[0:2],
                    device=visual_embeds.device,
                    dtype=torch.long,
                )
        v_out = self.falbert(
            frames=images,
            frames_mask=images_mask,
            visual_embeds=visual_embeds,
            mode="v",
        )
        return v_out

    def encode_text(
        self,
        input_ids: torch.Tensor,
        input_mask: torch.Tensor,
        input_segment_ids: torch.Tensor = None,
    ):
        if input_segment_ids is None:
            input_segment_ids = torch.zeros_like(input_ids, device=input_ids.device)
        t_out = self.falbert(
            input_ids=input_ids,
            input_segment_ids=input_segment_ids,
            input_mask=input_mask,
            mode="t",
        )
        return t_out

    def encode_multimodal(
        self,
        input_ids,
        input_segment_ids,
        input_mask,
        images=None,
        images_mask=None,
        visual_embeds=None,
        *args,
        **kwargs,
    ):
        if images_mask is None:
            if visual_embeds is None:
                images_mask = torch.ones(images.shape[0:2], device=images.device, dtype=torch.long)
            else:
                images_mask = torch.ones(
                    visual_embeds.shape[0:2],
                    device=visual_embeds.device,
                    dtype=torch.long,
                )
        mmout = self.falbert(
            input_ids=input_ids,
            input_segment_ids=input_segment_ids,
            input_mask=input_mask,
            frames=images,
            frames_mask=images_mask,
            visual_embeds=visual_embeds,
            mode="tv",
        )
        return mmout

    def init_projector(
        self,
        input_size=768,
        output_size=128,
    ):
        # projector = torch.nn.Linear(input_size, output_size)
        # v_projector = projector
        # t_projector = projector
        v_projector = torch.nn.Linear(input_size, output_size)
        t_projector = torch.nn.Linear(input_size, output_size)
        return t_projector, v_projector

    def cal_cls_loss(self, **kwargs):
        for key in ["out_lvl1", "label"]:
            if key in kwargs:
                kwargs[key] = self.all_gather(kwargs[key].contiguous())
                kwargs[key] = kwargs[key].flatten(0, 1)
        loss = cross_entropy(kwargs["out_lvl1"], kwargs["label"])
        return {"loss": loss}

    def training_step(self, batch, idx):
        token_ids, segment_ids, attn_mask, image, image_mask = (
            batch["input_ids"],
            batch["input_segment_ids"],
            batch["input_mask"],
            batch["frames"],
            batch["frames_mask"],
        )
        rep_dict = self.forward(
            input_ids=token_ids,
            input_segment_ids=segment_ids,
            input_mask=attn_mask,
            frames=image,
            frames_mask=image_mask,
        )
        rep_dict.update({"label": batch["label"]})

        loss_dict = self.cal_cls_loss(**rep_dict)

        acc, _, _ = accuracy(rep_dict["out_lvl1"], rep_dict["label"])
        gt = torch.eq(rep_dict["label"], 2).int()
        pred_score = self.softmax(rep_dict["out_lvl1"])[:, 2]
        loss_dict.update({"acc": acc, "b_pred": pred_score, "b_gt": gt})
        loss_dict.update({"learning_rate": self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]})

        return loss_dict

    def validation_step(self, batch, idx):
        return self.training_step(batch, idx)

    def validation_epoch_end(self, outputs) -> None:
        gathered_results = DIST_ENV.all_gather_object(outputs)
        all_results = []
        for item in gathered_results:
            all_results.extend(item)
        acc_all = [out["acc"] for out in all_results]
        total_acc = sum(acc_all) / len(acc_all)
        self.log("total_val_acc", total_acc, console=True)
        labels, scores = [], []
        for out in all_results:
            labels.extend(out["b_gt"].detach().cpu().tolist())
            scores.extend(out["b_pred"].detach().cpu().tolist())
        auc = roc_auc_score(labels, scores)
        precision, recall, thr = p_fix_r(np.array(scores), np.array(labels), 0.3)
        ###优质PR, auc
        self.log("total_val_prec", precision, console=True)
        self.log("total_val_rec", recall, console=True)
        self.log("total_val_auc", auc, console=True)

    def predict_step(self, batch, idx):
        token_ids, segment_ids, attn_mask, image, image_mask = (
            batch["input_ids"],
            batch["input_segment_ids"],
            batch["input_mask"],
            batch["frames"],
            batch["frames_mask"],
        )
        rep_dict = self.forward(
            input_ids=token_ids,
            input_segment_ids=segment_ids,
            input_mask=attn_mask,
            frames=image,
            frames_mask=image_mask,
        )
        return {
            "pred": self.softmax(rep_dict["out_lvl1"]),
            "label": batch["label"],
            "item_id": batch["item_id"],
        }

    def configure_optimizers(self):
        no_decay = ["bias", "bn", "norm"]
        no_dacay_params_dict = {"params": [], "weight_decay": 0.0}
        low_lr_params_dict = {
            "params": [],
            "weight_decay": self.config_optim.weight_decay,
            "lr": self.config_optim.learning_rate * 0.1,
        }
        normal_params_dict = {
            "params": [],
            "weight_decay": self.config_optim.weight_decay,
        }

        for n, p in self.named_parameters():
            if any(nd in n for nd in no_decay):
                no_dacay_params_dict["params"].append(p)
            elif n.startswith("albert"):
                low_lr_params_dict["params"].append(p)
            else:
                normal_params_dict["params"].append(p)
        optimizer_grouped_parameters = [
            no_dacay_params_dict,
            low_lr_params_dict,
            normal_params_dict,
        ]

        if self.config_optim.optim == "SGD":
            optm = torch.optim.SGD(
                optimizer_grouped_parameters,
                self.config_optim.learning_rate,
                momentum=self.config_optim.momentum,
                weight_decay=self.config_optim.weight_decay,
            )
        elif self.config_optim.optim == "AdamW":
            optm = AdamW(
                optimizer_grouped_parameters,
                lr=self.config_optim.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-6,
                weight_decay=self.config_optim.weight_decay,
                correct_bias=False,
            )

        if self.config_optim.lr_schedule == "linear":
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=optm,
                num_warmup_steps=self.config_optim.warmup_steps_factor * self.trainer.steps_per_epoch,
                num_training_steps=self.trainer.total_steps,
            )
        elif self.config_optim.lr_schedule == "cosine":
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optm,
                num_warmup_steps=self.config_optim.warmup_steps_factor * self.trainer.steps_per_epoch,
                num_training_steps=self.trainer.total_steps,
            )
        elif self.config_optim.lr_schedule == "onecycle":
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optm,
                max_lr=self.config_optim.learning_rate,
                total_steps=self.trainer.total_steps,
            )

        return {"optimizer": optm, "lr_scheduler": lr_scheduler}
