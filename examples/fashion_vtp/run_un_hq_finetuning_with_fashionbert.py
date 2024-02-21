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

# add metric
from sklearn.metrics import roc_auc_score

from easyguard.appzoo.fashion_vtp.model_finetune import FashionVTP
from easyguard.appzoo.fashion_vtp.un_hq_data_with_product import UnHqDataModule
from easyguard.appzoo.fashion_vtp.utils import compute_accuracy, p_fix_r, print_res, r_fix_p
from easyguard.core import AutoModel

# optimizer
from easyguard.core.lr_scheduler import build_scheduler
from easyguard.utils.arguments import print_cfg
from easyguard.utils.losses import cross_entropy


class MMClsModel(CruiseModule):
    def __init__(
        self,
        config_model: str = "./examples/fashion_vtp/un_hq_configs/config_model.yaml",
        config_optim: str = "./examples/fashion_vtp/un_hq_configs/config_optim.yaml",
        load_pretrained: str = "hdfs://haruna/home/byte_ecom_govern/user/yangmin.priv/weights/fashionvtp3-model-20230109.th",
    ):
        super(MMClsModel, self).__init__()
        self.save_hparams()

    def setup(self, stage) -> None:
        """
        Load yaml file as config class
        """
        with open(self.hparams.config_model) as fp:
            self.config_model = SimpleNamespace(**yaml.load(fp, yaml.Loader))
        # with hopen(self.hparams.config_optim) as fp:
        with open(self.hparams.config_optim) as fp:
            self.config_optim = SimpleNamespace(**yaml.load(fp, yaml.Loader))

        """
        Initialize modules
        """
        self.text_visual_and_fuse_model = FashionVTP(self.config_model)
        self.fashion_bert_model = AutoModel.from_pretrained("fashionbert-base")

        # self.reduc = nn.Linear(768 * 3, 768)
        self.reduc = nn.Linear(768 + 512, 768)
        self.classifier = torch.nn.Linear(768, self.config_optim.class_num)

        """
        Initialize loss
        """
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        """
        Initialize classifier weights and load pretrained
        """
        # self.init_weights()
        self.initialize_weights()
        self.freeze_params(self.config_optim.freeze_prefix)

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

    def initialize_weights(self):
        if hexists(self.hparams.load_pretrained):
            state_dict_ori = self.state_dict()
            state_dict_pretrain = load(self.hparams.load_pretrained, map_location="cpu")
            state_dict = {"text_visual_and_fuse_model." + k: v for k, v in state_dict_pretrain.items()}

            state_dict_new = OrderedDict()
            for key, value in state_dict.items():
                if key in state_dict_ori and state_dict_ori[key].shape == state_dict[key].shape:
                    state_dict_new[key] = value
            info = self.load_state_dict(state_dict_new, strict=False)
            # print("load info: ", info, file=sys.stderr)

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
        product_ids,
        product_mask,
        product_segment_ids,
        frames,
        frames_mask,
        product_images,
        product_images_mask,
        labels,
        *args,
        **kwargs,
    ):
        mm_emb, t_emb, v_emb = self.text_visual_and_fuse_model(
            input_ids=input_ids,
            input_segment_ids=input_segment_ids,
            input_mask=input_mask,
            images=frames,
            images_mask=frames_mask,
        )

        # product_i_out = self.text_visual_and_fuse_model.encode_image(images=product_images, images_mask=product_images_mask)
        # product_t_out = self.text_visual_and_fuse_model.encode_text(input_ids=product_ids, input_segment_ids=product_segment_ids, input_mask=product_mask)

        # product_t_emb = product_t_out['pooled_output']
        # product_i_emb = product_i_out['pooled_output']

        # product_emb = torch.cat((product_t_emb, product_i_emb), dim=-1)

        b, t, c, h, w = product_images.size()
        product_image = product_images.view(b * t, c, h, w)

        product_emb = self.fashion_bert_model(product_ids, product_image)["fuse_rep"]  # 512
        # print("product_emb: ", product_emb.size())

        emb = torch.cat((mm_emb, product_emb), dim=-1)

        final_emb = self.reduc(emb)
        logits = self.classifier(final_emb)

        return {"logits": logits}

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

        (
            product_ids,
            product_segment_ids,
            product_mask,
            product_image,
            product_image_mask,
        ) = (
            batch["product_ids"],
            batch["product_segment_ids"],
            batch["product_mask"],
            batch["product_images"],
            batch["product_images_mask"],
        )

        rep_dict = self.forward(
            input_ids=token_ids,
            input_segment_ids=segment_ids,
            input_mask=attn_mask,
            product_ids=product_ids,
            product_mask=product_mask,
            product_segment_ids=product_segment_ids,
            frames=image,
            frames_mask=image_mask,
            product_images=product_image,
            product_images_mask=product_image_mask,
            labels=batch["labels"],
        )
        rep_dict.update({"labels": batch["labels"]})
        cls_loss = self.cal_cls_loss(**rep_dict)
        acc = compute_accuracy(rep_dict["logits"], rep_dict["labels"])

        train_loss_dict = {"loss": cls_loss, "train_acc": acc}
        return train_loss_dict

    def validation_step(self, batch, idx):
        token_ids, segment_ids, attn_mask, image, image_mask = (
            batch["input_ids"],
            batch["input_segment_ids"],
            batch["input_mask"],
            batch["frames"],
            batch["frames_mask"],
        )

        (
            product_ids,
            product_segment_ids,
            product_mask,
            product_image,
            product_image_mask,
        ) = (
            batch["product_ids"],
            batch["product_segment_ids"],
            batch["product_mask"],
            batch["product_images"],
            batch["product_images_mask"],
        )

        val_rep_dict = self.forward(
            input_ids=token_ids,
            input_segment_ids=segment_ids,
            input_mask=attn_mask,
            product_ids=product_ids,
            product_mask=product_mask,
            product_segment_ids=product_segment_ids,
            frames=image,
            frames_mask=image_mask,
            product_images=product_image,
            product_images_mask=product_image_mask,
            labels=batch["labels"],
        )
        val_rep_dict.update({"labels": batch["labels"]})
        val_loss = self.cal_cls_loss(**val_rep_dict)
        # val_loss = self.cal_arc_loss(**val_rep_dict)

        val_loss_dict = {"val_loss": val_loss}
        # compute metric
        gt = torch.eq(val_rep_dict["labels"], 0).int()
        pred_score = F.softmax(val_rep_dict["logits"], dim=1)[:, 0]

        val_loss_dict.update({"scores": pred_score, "gts": gt})
        val_loss_dict.update(
            {
                "logits": val_rep_dict["logits"],
                "labels": val_rep_dict["labels"],
            }
        )

        return val_loss_dict

    def cal_cls_loss(self, **kwargs):
        for key in ["logits", "labels"]:
            if key in kwargs:
                kwargs[key] = self.all_gather(kwargs[key].contiguous())
                kwargs[key] = kwargs[key].flatten(0, 1)
        loss = self.criterion(kwargs["logits"], kwargs["labels"])
        return loss

    def cal_arc_loss(self, **kwargs):
        for key in ["arc_logits", "labels"]:
            if key in kwargs:
                kwargs[key] = self.all_gather(kwargs[key].contiguous())
                kwargs[key] = kwargs[key].flatten(0, 1)
        arc_loss = cross_entropy(kwargs["arc_logits"], kwargs["labels"])
        return arc_loss

    @torch.no_grad()
    def validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        """
        对validation的所有step的结果进行处理
        """

        """
        gathered_results = DIST_ENV.all_gather_object(outputs)
        all_results = []
        for item in gathered_results:
            all_results.extend(item)
        scores = []
        labels = []
        for out in all_results:
            scores.extend(out["scores"].detach().cpu().tolist())
            labels.extend(out["gts"].detach().cpu().tolist())

        auc = roc_auc_score(np.asarray(labels, dtype=int), np.asarray(scores, dtype=float))
        precision, recall, thr = p_fix_r(np.asarray(scores, dtype=float), np.asarray(labels, dtype=int), 0.3)

        val_metric_dict = {'auc': auc, 'precision': precision, 'recall': recall}
        self.log_dict(val_metric_dict, console=True)
        self.log("precision", precision, console=True)
        self.log("recall", recall, console=True)
        self.log("auc", auc, console=True)
        """

        gathered_results = DIST_ENV.all_gather_object(outputs)
        all_results = []
        for item in gathered_results:
            all_results.extend(item)
        all_logits = []
        all_labels = []
        all_scores = []
        all_gts = []

        for out in all_results:
            all_logits.extend(out["logits"].detach().cpu().tolist())
            all_labels.extend(out["labels"].detach().cpu().tolist())
            all_scores.extend(out["scores"].detach().cpu().tolist())
            all_gts.extend(out["gts"].detach().cpu().tolist())

        logits = torch.from_numpy(np.array(all_logits))
        labels = torch.from_numpy(np.array(all_labels))

        accuracy = compute_accuracy(logits, labels)

        auc = roc_auc_score(np.asarray(all_gts, dtype=int), np.asarray(all_scores, dtype=float))
        precision, recall, thr = r_fix_p(
            np.asarray(all_scores, dtype=float),
            np.asarray(all_gts, dtype=int),
            0.7,
        )

        val_metric_dict = {
            "val_acc": accuracy,
            "auc": auc,
            "precision": precision,
            "recall": recall,
        }
        self.log_dict(val_metric_dict, console=True)

    def predict_step(self, batch, idx):
        token_ids, segment_ids, attn_mask, image, image_mask = (
            batch["input_ids"],
            batch["input_segment_ids"],
            batch["input_mask"],
            batch["frames"],
            batch["frames_mask"],
        )

        (
            product_ids,
            product_segment_ids,
            product_mask,
            product_image,
            product_image_mask,
        ) = (
            batch["product_ids"],
            batch["product_segment_ids"],
            batch["product_mask"],
            batch["product_images"],
            batch["product_images_mask"],
        )

        rep_dict = self.forward(
            input_ids=token_ids,
            input_segment_ids=segment_ids,
            input_mask=attn_mask,
            product_ids=product_ids,
            product_mask=product_mask,
            product_segment_ids=product_segment_ids,
            frames=image,
            frames_mask=image_mask,
            product_images=product_image,
            product_images_mask=product_image_mask,
            labels=batch["labels"],
        )
        # compute metric
        # gt = torch.eq(rep_dict["labels"], 2).int()
        # pred_score = F.softmax(rep_dict["logits"], dim=1)[:, 2]
        return {
            "pred": F.softmax(rep_dict["logits"]),
            "label": batch["labels"],
            "object_id": batch["object_ids"],
        }

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
            elif n.startswith("text_visual_and_fuse_model") or n.startswith("fashion_bert_model"):
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
        MMClsModel,
        trainer_class=CruiseTrainer,
        datamodule_class=UnHqDataModule,
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
            "default_root_dir": "/mnt/bn/ecom-tianke-lq/cruise_train",
            "default_hdfs_dir": None
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
    """
    datamodule.local_rank_zero_prepare()
    datamodule.setup()
    # model.partial_load_from_checkpoints('/mnt/bn/ecom-tianke-lq/cruise_train/version_1/checkpoints/epoch=14-step=25000-loss=0.193.ckpt', rename_params={})
    outputs = trainer.predict(model,
                            predict_dataloader=datamodule.predict_dataloader(),
                            sync_predictions=True)

    print('================= 2fen =================')
    label = []
    score = []
    for out in outputs:
        pred = out['pred'].cpu().numpy()
        grt = out['label'].cpu().numpy()
        for i in range(pred.shape[0]):
            score.append(pred[i, 0])
            if int(grt[i]) == 0:
                label.append(1)
            else:
                label.append(0)
    print(len(label),len(score),sum(label))
    print('auc: ',roc_auc_score(label,score))
    print_res(np.array(score),np.array(label))

    fw = open("/mnt/bn/ecom-tianke-lq/cruise_train/version_1/predict_result.txt", 'w')
    for out in outputs:
        pred = out['pred'].cpu().numpy()
        grt = out['label'].cpu().numpy()
        item_ids = out['object_id']
        for i in range(pred.shape[0]):
            fw.write('{} {} {} {} {}\n'.format(str(item_ids[i]), pred[i, 0], pred[i, 1], pred[i, 2], grt[i]))

    """
