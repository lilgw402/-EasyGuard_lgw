# -*- coding: utf-8 -*-


from typing import Optional, Tuple

import numpy as np
import torch
from transformers import AutoModel

try:
    import cruise  # noqa: F401
    from cruise import CruiseModule
except ImportError:
    print(
        "[ERROR] cruise is not installed! Please refer this doc: https://bytedance.feishu.cn/wiki/wikcnGP7yzZAuKpPfL6jRJKl2ag"  # noqa: E501
    )

from easyguard.core.optimizers import build_optimizer, build_scheduler
from easyguard.modelzoo.modeling_utils import load_pretrained
from easyguard.utils.losses import cross_entropy


class ModelZooTextClf(CruiseModule):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "bert",
        #  cl_enable: bool = False,
        #  cl_temp: float = 0.05,
        #  cl_weight: float = 1.0,
        #  ntx_enable: bool = False,
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

        # if pretrained_model_name_or_path == "fashionxlm-mdeberta-v3-base":
        #     from ...modelzoo.models.mdeberta_v2 import DebertaV2ForMaskedLM
        #     self.backbone = DebertaV2ForMaskedLM.from_pretrained('microsoft/mdeberta-v3-base')
        # else:
        #     self.backbone = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path)
        self.backbone = AutoModel.from_pretrained(pretrained_model_name_or_path)

        # use classification learning loss
        self.classification_task_head = classification_task_head
        self.classifier = torch.nn.Linear(hidden_size, self.classification_task_head)

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

    def freeze_params(self, freeze_prefix):
        for name, param in self.named_parameters():
            for prefix in freeze_prefix:
                if name.startswith(prefix):
                    param.requires_grad = False

    def rank_zero_prepare(self):
        # load partial pretrain
        if self.hparams.load_pretrain:
            load_pretrained(self.hparams.load_pretrain, self)

    def forward(self, input_ids, token_type_ids, attention_mask, labels):
        """
        input_ids: [bsz, seq_len]
        input_segment_ids: [bsz, seq_len]
        input_mask: [bsz, seq_len]
        labels: [bsz]
        """
        output_dict = {}

        # classification task
        # mmout = self.backbone(input_ids, attention_mask, token_type_ids,
        #                       labels=None, output_hidden_states=True)
        mmout = self.backbone(input_ids, attention_mask, token_type_ids)

        # hidden_states = mmout.hidden_states[-1]  # batch * sen_len * emd_size
        # cls_status = hidden_states[:, 0, :]  # batch * emd_size
        cls_status = mmout.pooler_output
        logits = self.classifier(cls_status)  # batch * label_size
        loss = cross_entropy(logits, labels)

        output_dict["loss"] = loss

        # collect results for validation
        output_dict["diff"] = (labels.long() == torch.argmax(logits, 1).long()).float()
        # output_dict['predictions'] = torch.argmax(logits, 1).float()

        return output_dict

    def training_step(self, batch, idx):
        return self(**batch)

    def validation_step(self, batch, idx):
        return self(**batch)

    def validation_epoch_end(self, outputs) -> None:
        # compute validation results at val_check_interval
        # TODO: need to apply all_gather op for distributed training (multiple workers)
        all_labels = []
        # all_predictions = []
        for out in outputs:
            all_labels.extend(out["diff"])
        all_labels = np.array(all_labels).reshape([-1])
        acc_score = np.average(all_labels)
        self.log("acc_score", acc_score, console=True)
        print("shape", all_labels.shape)
        print("sum", np.sum(all_labels))
        print("acc_score", acc_score)

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
