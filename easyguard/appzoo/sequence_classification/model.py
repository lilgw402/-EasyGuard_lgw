# -*- coding: utf-8 -*-

import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from easyguard import AutoModel

try:
    import cruise
    from cruise import CruiseModule
except ImportError:
    print(
        "[ERROR] cruise is not installed! Please refer this doc: https://bytedance.feishu.cn/wiki/wikcnGP7yzZAuKpPfL6jRJKl2ag"
    )

from ...core.optimizers import build_optimizer, build_scheduler
from ...utils import logging
from ...utils.losses import LearnableNTXentLoss, cross_entropy

logger = logging.get_logger(__name__)


class SequenceClassificationModel(CruiseModule):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "fashionxlm-base",
        classification_task_enable: bool = False,
        classification_task_head: int = 2,
        hidden_size: int = 768,
        load_pretrain: Optional[str] = None,
        all_gather_limit: int = -1,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.05,
        base_lr: float = 5e-4,
        warmup_lr: float = 5e-7,
        min_lr: float = 5e-6,
        lr_scheduler: str = "cosine",
        lr_scheduler_decay_ratio: float = 0.8,
        lr_scheduler_decay_rate: float = 0.1,
        optimizer: str = "adamw",
        optimizer_eps: float = 1e-8,
        optimizer_betas: Tuple[float, ...] = [0.9, 0.999],
        momentum: float = 0.9,
    ):
        super().__init__()
        self.save_hparams()
        self.model = AutoModel.from_pretrained(
            pretrained_model_name_or_path,
            model_cls="sequence_model",
            num_labels=2,
        )

    def forward(self, **kwargs):
        """
        input_ids: [bsz, seq_len]
        input_segment_ids: [bsz, seq_len]
        input_mask: [bsz, seq_len]
        labels: [bsz]
        """
        output_dict = {}
        labels = kwargs.get("labels")

        language_key_save_model_names = ["fashionxlm_moe"]
        # classification task
        # remove the language key if model_type not in `language_key_save_model_names`
        if self.model.extra_args["model_type"] not in language_key_save_model_names:
            kwargs.pop("language")
        mmout = self.model(**kwargs)

        output_dict["loss"] = mmout.loss
        output_dict["predict"] = mmout.logits
        output_dict["lables"] = labels
        output_dict["diff"] = (labels.long() == torch.argmax(mmout.logits, 1).long()).float()

        return output_dict

    def training_step(self, batch, idx):
        return self(**batch)

    def validation_step(self, batch, idx):
        return self(**batch)

    def validation_epoch_end(self, outputs) -> None:
        # compute validation results at val_check_interval
        # TODO: need to apply all_gather op for distributed training (multiple workers)
        from sklearn.metrics import f1_score

        all_diff = []
        all_predictions = []
        all_labels_true = []
        for out in outputs:
            all_diff.extend(out["diff"])
            all_predictions.extend(torch.argmax(out["predict"], 1).long())
            all_labels_true.extend(out["lables"].long())

        all_labels = np.array(all_diff).reshape([-1])
        acc_score = np.average(all_labels)
        true_labels = np.array(all_labels_true).reshape([-1])
        predict = np.array(all_predictions).reshape([-1])
        f1 = f1_score(true_labels, predict, average="micro")
        logger.info(f"acc score: {acc_score}")
        logger.info(f"f1 score: {f1}")
        logger.info(f"shape: all_labels.shape")
        logger.info(f"sum: {np.sum(all_labels)}")

    def configure_optimizers(self):
        optimizer = build_optimizer(self.hparams, self)
        lr_scheduler = build_scheduler(
            self.hparams,
            optimizer,
            self.trainer.max_epochs,
            self.trainer.steps_per_epoch // self.trainer._accumulate_grad_batches,
        )
        return [optimizer], [lr_scheduler]

    def lr_scheduler_step(self, schedulers, **kwargs):
        # timm lr scheduler is called every step
        for scheduler in schedulers:
            scheduler.step_update(self.trainer.global_step // self.trainer._accumulate_grad_batches)

    def on_fit_start(self) -> None:
        self.rank_zero_print("===========My custom fit start function is called!============")
