# -*- coding: utf-8 -*-
import os
import os.path
import random
import sys
from collections import OrderedDict
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from cruise import CruiseCLI, CruiseModule, CruiseTrainer
from cruise.trainer.callback import EarlyStopping
from cruise.utilities.cloud_io import load
from cruise.utilities.distributed import DIST_ENV
from cruise.utilities.hdfs_io import hexists, hopen
from torch import optim

try:
    import easyguard
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from live_site_data import LiveSiteDataModule

# add metric
from sklearn.metrics import roc_auc_score

from easyguard.appzoo.fashion_vtp.utils import compute_accuracy, compute_f1_score

# model
from easyguard.core import AutoModel

# optimizer
from easyguard.core.lr_scheduler import build_scheduler
from easyguard.utils.arguments import print_cfg
from easyguard.utils.losses import cross_entropy


class MyModel(CruiseModule):
    def __init__(
        self,
        config_optim: str = "./examples/fashion_vtp/live_site_configs/config_optim.yaml",
    ):
        super(MyModel, self).__init__()
        self.save_hparams()

    def setup(self, stage) -> None:
        """
        Load yaml file as config class
        """
        with open(self.hparams.config_optim) as fp:
            self.config_optim = SimpleNamespace(**yaml.load(fp, yaml.Loader))

        self.fashionvtp_model = AutoModel.from_pretrained("fashionvtp-base-c")

        self.classifier_1 = nn.Linear(768, self.config_optim.class_num_lv1)
        self.classifier_2 = nn.Linear(768, self.config_optim.class_num_lv2)

        self.criterion_1 = nn.CrossEntropyLoss(ignore_index=-1)
        self.criterion_2 = nn.CrossEntropyLoss(ignore_index=-1)

        self.freeze_params(self.config_optim.freeze_prefix)

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
        frames,
        frames_mask,
        *args,
        **kwargs,
    ):
        mm_emb, t_emb, v_emb = self.fashionvtp_model(
            input_ids=input_ids,
            input_segment_ids=input_segment_ids,
            input_mask=input_mask,
            images=frames,
            images_mask=frames_mask,
        )

        logits_1 = self.classifier_1(mm_emb)
        logits_2 = self.classifier_2(mm_emb)
        return {"logits_1": logits_1, "logits_2": logits_2}

    def on_train_epoch_start(self):
        self.trainer.train_dataloader._loader.sampler.set_epoch(self.trainer.current_epoch)

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
        rep_dict.update({"labels_1": batch["labels_1"]})
        cls_loss_lv1 = self.cal_cls_loss_v1(keys=["logits_1", "labels_1"], **rep_dict)

        rep_dict.update({"labels_2": batch["labels_2"]})
        cls_loss_lv2 = self.cal_cls_loss_v2(keys=["logits_2", "labels_2"], **rep_dict)

        train_loss = cls_loss_lv1 + cls_loss_lv2

        acc1 = compute_accuracy(rep_dict["logits_1"], rep_dict["labels_1"])
        acc2 = compute_accuracy(rep_dict["logits_2"], rep_dict["labels_2"])

        train_loss_dict = {
            "loss": train_loss,
            "cls_loss1": cls_loss_lv1,
            "cls_loss2": cls_loss_lv2,
            "train_acc1": acc1,
            "train_acc_2": acc2,
        }
        return train_loss_dict

    def validation_step(self, batch, idx):
        token_ids, segment_ids, attn_mask, image, image_mask = (
            batch["input_ids"],
            batch["input_segment_ids"],
            batch["input_mask"],
            batch["frames"],
            batch["frames_mask"],
        )

        val_rep_dict = self.forward(
            input_ids=token_ids,
            input_segment_ids=segment_ids,
            input_mask=attn_mask,
            frames=image,
            frames_mask=image_mask,
        )
        val_rep_dict.update({"labels_1": batch["labels_1"]})
        val_cls_loss_lv1 = self.cal_cls_loss_v1(keys=["logits_1", "labels_1"], **val_rep_dict)

        val_rep_dict.update({"labels_2": batch["labels_2"]})
        val_cls_loss_lv2 = self.cal_cls_loss_v2(keys=["logits_2", "labels_2"], **val_rep_dict)

        val_loss = val_cls_loss_lv1 + 2.0 * val_cls_loss_lv2
        val_loss_dict = {"loss": val_loss}
        val_loss_dict.update(
            {
                "logits_1": val_rep_dict["logits_1"],
                "labels_1": val_rep_dict["labels_1"],
            }
        )
        val_loss_dict.update(
            {
                "logits_2": val_rep_dict["logits_2"],
                "labels_2": val_rep_dict["labels_2"],
            }
        )
        return val_loss_dict

    def cal_cls_loss_v1(self, keys, **kwargs):
        for key in keys:
            if key in kwargs:
                kwargs[key] = self.all_gather(kwargs[key].contiguous())
                kwargs[key] = kwargs[key].flatten(0, 1)
        loss = self.criterion_1(kwargs[keys[0]], kwargs[keys[1]])

        return loss

    def cal_cls_loss_v2(self, keys, **kwargs):
        for key in keys:
            if key in kwargs:
                kwargs[key] = self.all_gather(kwargs[key].contiguous())
                kwargs[key] = kwargs[key].flatten(0, 1)
        loss = self.criterion_2(kwargs[keys[0]], kwargs[keys[1]])
        return loss

    @torch.no_grad()
    def validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        """
        对validation的所有step的结果进行处理
        """
        gathered_results = DIST_ENV.all_gather_object(outputs)
        all_results = []
        for item in gathered_results:
            all_results.extend(item)
        all_logits_1 = []
        all_labels_1 = []
        all_logits_2 = []
        all_labels_2 = []

        for out in all_results:
            all_logits_1.extend(out["logits_1"].detach().cpu().tolist())
            all_logits_2.extend(out["logits_2"].detach().cpu().tolist())
            all_labels_1.extend(out["labels_1"].detach().cpu().tolist())
            all_labels_2.extend(out["labels_2"].detach().cpu().tolist())

        logits_1 = torch.from_numpy(np.array(all_logits_1))
        labels_1 = torch.from_numpy(np.array(all_labels_1))
        logits_2 = torch.from_numpy(np.array(all_logits_2))
        labels_2 = torch.from_numpy(np.array(all_labels_2))

        macro_f1_lv1 = compute_f1_score(logits_1, labels_1)
        macro_f1_lv2 = compute_f1_score(logits_2, labels_2)
        accuracy_lv1 = compute_accuracy(logits_1, labels_1)
        accuracy_lv2 = compute_accuracy(logits_2, labels_2)

        val_metric_dict = {
            "macro_f1_lv1": macro_f1_lv1,
            "macro_f1_lv2": macro_f1_lv2,
            "val_acc1": accuracy_lv1,
            "val_acc2": accuracy_lv2,
        }
        self.log_dict(val_metric_dict, console=True)
        self.log("val_acc1", accuracy_lv1, console=True)
        self.log("val_acc2", accuracy_lv2, console=True)
        self.log("macro_f1_lv1", macro_f1_lv1, console=True)
        self.log("macro_f1_lv2", macro_f1_lv2, console=True)

    def configure_optimizers(self):
        no_decay = ["bias", "bn", "norm"]
        no_dacay_params_dict = {"params": [], "weight_decay": 0.0}
        low_lr_params_dict = {
            "params": [],
            "weight_decay": self.config_optim.weight_decay,
            "lr": self.config_optim.base_lr * 0.1,
        }
        normal_params_dict = {
            "params": [],
            "weight_decay": self.config_optim.weight_decay,
            "lr": self.config_optim.base_lr,
        }

        for n, p in self.named_parameters():
            if any(nd in n for nd in no_decay):
                no_dacay_params_dict["params"].append(p)
            elif n.startswith("fashionvtp_model"):
                low_lr_params_dict["params"].append(p)
            else:
                normal_params_dict["params"].append(p)

        optimizer_grouped_parameters = [
            no_dacay_params_dict,
            low_lr_params_dict,
            normal_params_dict,
        ]

        if self.config_optim.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                optimizer_grouped_parameters,
                momentum=self.config_optim.momentum,
                nesterov=True,
                lr=self.config_optim.base_lr,
                weight_decay=self.config_optim.weight_decay,
            )

        elif self.config_optim.optimizer == "adamw":
            optimizer = optim.AdamW(
                optimizer_grouped_parameters,
                eps=self.config_optim.optimizer_eps,
                betas=(0.9, 0.999),
                lr=self.config_optim.base_lr,
                weight_decay=self.config_optim.weight_decay,
            )

        lr_scheduler = build_scheduler(
            self.config_optim,
            optimizer,
            self.trainer.max_epochs,
            self.trainer.steps_per_epoch // self.trainer._accumulate_grad_batches,
        )
        return [optimizer], [lr_scheduler]

    def lr_scheduler_step(self, schedulers, **kwargs):
        # timm lr scheduler is called every step
        for scheduler in schedulers:
            scheduler.step_update(self.trainer.global_step // self.trainer._accumulate_grad_batches)


if __name__ == "__main__":
    cli = CruiseCLI(
        MyModel,
        trainer_class=CruiseTrainer,
        datamodule_class=LiveSiteDataModule,
        trainer_defaults={
            "precision": 16,
            "log_every_n_steps": 100,
            "max_epochs": 30,
            "enable_versions": True,
            "val_check_interval": [1000, 1.0],
            "sync_batchnorm": True,
            "find_unused_parameters": True,
            "summarize_model_depth": 5,
            "checkpoint_monitor": "loss",
            "checkpoint_mode": "min",
            # 'callbacks': [EarlyStopping(monitor='precision',
            #                             mode='max',
            #                             min_delta=0.001,
            #                             patience=2,
            #                             verbose=True,
            #                             stopping_threshold=0.5)]
        },
    )
    cfg, trainer, model, datamodule = cli.parse_args()
    print_cfg(cfg)
    trainer.fit(model, datamodule)
