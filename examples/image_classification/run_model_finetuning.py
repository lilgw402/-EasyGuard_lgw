# --------------------------------------------------------
# AutoModel Finetune
# Swin-Transformer
# Copyright (c) 2023 EasyGuard
# Written by yangmin.priv
# --------------------------------------------------------

import os
import sys
from collections import OrderedDict
from types import SimpleNamespace
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import yaml
from PIL import Image

try:
    import easyguard
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cruise import CruiseCLI, CruiseModule, CruiseTrainer

from easyguard import AutoModel
from easyguard.core.lr_scheduler import build_scheduler

# from cruise.data_module.byted_data_factory.imagenet import BytedImageNetDataModule
from examples.image_classification.data import MyDataModule


class SimpleModel(CruiseModule):
    def __init__(
        self,
        model_arch: str = "fashion-swin-base-224-fashionvtp",
        config_optim: str = "./examples/image_classification/config_optim.yaml",
    ):
        super().__init__()
        self.save_hparams()

    def setup(self, stage) -> None:
        with open(self.hparams.config_optim) as fp:
            self.config_optim = SimpleNamespace(**yaml.load(fp, yaml.Loader))
        self.model = AutoModel.from_pretrained(self.hparams.model_arch)
        self.model.head = nn.Linear(1024, self.config_optim.class_num)

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        # x, y = batch['image']['data'], batch['label']
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        y_pred = torch.argmax(torch.softmax(y_hat, dim=-1), dim=-1)
        train_acc = torch.sum(y_pred.long() == y.long()).float() / y.numel()
        self.log("train_acc", train_acc, console=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # x, y = batch['image']['data'], batch['label']
        y_hat = torch.argmax(torch.softmax(self(x), dim=-1), dim=-1)
        val_acc = torch.sum((y_hat.long()) == y.long()).float() / y.numel()
        self.log("val_acc", val_acc)
        return {"val_acc": val_acc}

    def configure_optimizers(self):
        """
        -对params做特殊处理，[no_decay, low_lr_params, normal_params]
        -返回自定义的optimizer与lr_scheduler，这里主要是替换成了timm的实现
            optimizer: [sgd/adamw]
            lr_scheduler: [cosine/linear/step]
        """
        no_decay = ["bias", "bn", "norm"]
        no_dacay_params_dict = {"params": [], "weight_decay": 0.0}
        low_lr_params_dict = {
            "params": [],
            "weight_decay": self.config_optim.weight_decay,
            # "lr": self.config_optim.base_lr * 0.1, # here to set low lr parameters
            "lr": self.config_optim.base_lr,
        }
        normal_params_dict = {
            "params": [],
            "weight_decay": self.config_optim.weight_decay,
            "lr": self.config_optim.base_lr,
        }

        for n, p in self.named_parameters():
            if any(nd in n for nd in no_decay):
                no_dacay_params_dict["params"].append(p)
            elif n.startswith("model.head"):
                normal_params_dict["params"].append(p)
            else:
                low_lr_params_dict["params"].append(p)

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
            optimizer = torch.optim.AdamW(
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
        SimpleModel,
        trainer_class=CruiseTrainer,
        # datamodule_class=BytedImageNetDataModule,
        datamodule_class=MyDataModule,
        trainer_defaults={
            "max_epochs": 100,
            "val_check_interval": [1000, 1.0],
            "summarize_model_depth": 2,
            "checkpoint_monitor": "val_acc",
            "checkpoint_mode": "max",
        },
    )
    cfg, trainer, model, datamodule = cli.parse_args()
    trainer.fit(model, datamodule)
