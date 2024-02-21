# -*- coding: utf-8 -*-
"""
    @file run_fashionvtp_pretrain.py
    @brief
    @description FashionVTP3 pretrain demo
    @author yangmin.priv@bytedance.com
    @date: 2023/04/01 16:04:36
"""
import os
import os.path
import pdb
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
from cruise.utilities.cloud_io import load
from cruise.utilities.distributed import DIST_ENV
from cruise.utilities.hdfs_io import hexists, hopen
from torch import optim

try:
    import easyguard
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from easyguard.appzoo.fashion_vtp.data import ByteDriveDataModule

# appzoo中实现好自己的data模块
from easyguard.appzoo.fashion_vtp.data_pretrain import PretrainDataModule

# 导入FashionVTP模型，无特殊需求不需要改动，import进来即可
from easyguard.appzoo.fashion_vtp.model_pretrain import FashionVTP
from easyguard.appzoo.fashion_vtp.utils import compute_accuracy

# 加入了timm的lr_scheduler，在config_optim中指定，['cosine'/'linear'/'step']
from easyguard.core.lr_scheduler import build_scheduler
from easyguard.utils.arguments import print_cfg


class MMClsModel(CruiseModule):
    """
    初始化参数:
        config_optim: 训练策略相关配置
        config_model: 模型定义的配置
        load_pretrained: 预训练权重路径
    """

    def __init__(
        self,
        config_model: str = "./examples/fashion_vtp/pretrain_configs/config_model.yaml",
        config_optim: str = "./examples/fashion_vtp/pretrain_configs/config_optim.yaml",
        load_pretrained: str = "hdfs://haruna/home/byte_ecom_govern/user/yangmin.priv/1e_fvtp3_continue_pretrain_0220_8new/model_state_epoch_1508298.th",
    ):
        super(MMClsModel, self).__init__()
        self.save_hparams()

    def setup(self, stage) -> None:
        """
        Load yaml file as config class
        """
        # 从hdfs加载
        # with hopen(self.hparams.config_model) as fp:
        # 从本地加载
        with open(self.hparams.config_model) as fp:
            self.config_model = SimpleNamespace(**yaml.load(fp, yaml.Loader))
        # with hopen(self.hparams.config_optim) as fp:
        with open(self.hparams.config_optim) as fp:
            self.config_optim = SimpleNamespace(**yaml.load(fp, yaml.Loader))

        self.temp = nn.Parameter(torch.ones([]) * self.config_optim.temp)
        self.queue_size = self.config_optim.queue_size
        # self.momentum = self.config_optim.momentum_update
        self.momentum = 0.995
        self.alpha = self.config_optim.alpha

        """
        Initialize modules
            "text_visual_and_fuse_model": FashionVTP多模态backbone,定义后，model结构为
                                        {
                                            text_visual_and_fuse_model.falbert.xxx,
                                            text_visual_and_fuse_model.fusemodel.xxx,
                                            text_visual_and_fuse_model.xxx.xxx,
                                            ...
                                        }
            "classifier": 分类层
            "category_classifier": 分类层
        """
        self.text_visual_and_fuse_model = FashionVTP(self.config_model)

        self.model_pairs = [
            [
                self.text_visual_and_fuse_model.falbert,
                self.text_visual_and_fuse_model.falbert_m,
            ],
            [
                self.text_visual_and_fuse_model.t_projector,
                self.text_visual_and_fuse_model.t_projector_m,
            ],
            [
                self.text_visual_and_fuse_model.v_projector,
                self.text_visual_and_fuse_model.v_projector_m,
            ],
        ]

        # itm
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 2),
        )
        # just for demo(this data has 3 categories)
        self.category_classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 256),  # add a proj
            torch.nn.Linear(256, 130),
        )

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.category_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

        # create the queue
        self.register_buffer("image_queue", torch.randn(128, self.queue_size))
        self.register_buffer("text_queue", torch.randn(128, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        # 会有一些参数不在pretrain里，初始化它
        # self.init_weights()
        # 主要是load pretrain weights
        self.initialize_weights()
        self.copy_params()

        # test
        # model_pair = self.model_pairs[0]
        # for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
        #     print(param, param_m)

        # freeze不需要更新的参数
        self.freeze_params(self.config_optim.freeze_prefix)

    def init_weights(self):
        def init_weight_module(module):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.apply(init_weight_module)

    def initialize_weights(self):
        if hexists(self.hparams.load_pretrained):
            state_dict_ori = self.state_dict()
            state_dict_pretrain = load(self.hparams.load_pretrained, map_location="cpu")
            state_dict = {
                "text_visual_and_fuse_model." + k: v for k, v in state_dict_pretrain.items() if not "classifier" in k
            }

            # load the same classifer weights from ptrtrained if continue pretain on the same pretrain data
            """
            state_dict.update({'classifier.0.weight': state_dict_pretrain['classifier.0.weight'], 'classifier.0.bias': state_dict_pretrain['classifier.0.bias'],
                               'category_classifier.0.weight': state_dict_pretrain['category_classifier.0.weight'], 'category_classifier.0.bias': state_dict_pretrain['category_classifier.0.bias'],
                               'category_classifier.1.weight': state_dict_pretrain['category_classifier.1.weight'], 'category_classifier.1.bias': state_dict_pretrain['category_classifier.1.bias']})
            """
            state_dict_new = OrderedDict()
            for key, value in state_dict.items():
                if key in state_dict_ori and state_dict_ori[key].shape == state_dict[key].shape:
                    state_dict_new[key] = value
            info = self.load_state_dict(state_dict_new, strict=False)
            print("load info: ", info, file=sys.stderr)

    def freeze_params(self, freeze_prefix):
        for name, param in self.named_parameters():
            for prefix in freeze_prefix:
                if name.startswith(prefix):
                    param.requires_grad = False

    def forward(self, x):
        return self.text_visual_and_fuse_model(x)

    def training_step(self, batch, idx):
        input_ids, input_segment_ids, input_mask, frames, frames_mask = (
            batch["input_ids"],
            batch["input_segment_ids"],
            batch["input_mask"],
            batch["frames"],
            batch["frames_mask"],
        )

        # if self.trainer.current_epoch>0:
        #     alpha = self.alpha
        # else:
        #     alpha = self.alpha * min(1.0, self.trainer.current_step / self.trainer.steps_per_epoch)
        alpha = self.alpha

        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        t_output = self.text_visual_and_fuse_model.encode_text(
            input_ids=input_ids,
            input_segment_ids=input_segment_ids,
            input_mask=input_mask,
        )
        t_emb = t_output["pooled_output"]
        t_emb = self.text_visual_and_fuse_model.t_projector(t_emb)
        text_feat = F.normalize(t_emb, dim=-1)

        t_tokens = t_output["encoded_layers"][-1]

        v_output = self.text_visual_and_fuse_model.encode_image(images=frames, images_mask=frames_mask)
        v_emb = v_output["pooled_output"]
        v_emb = self.text_visual_and_fuse_model.v_projector(v_emb)
        image_feat = F.normalize(v_emb, dim=-1)

        v_tokens = v_output["encoded_layers"][-1][:, 1:, :]

        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            v_output_m = self.text_visual_and_fuse_model.encode_image_m(images=frames, images_mask=frames_mask)
            v_emb_m = v_output_m["pooled_output"]
            v_emb_m = self.text_visual_and_fuse_model.v_projector_m(v_emb_m)
            image_feat_m = F.normalize(v_emb_m, dim=-1)

            # v_tokens_m = v_output_m['encoded_layers'][-1][:, 1:, :]

            image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)

            t_output_m = self.text_visual_and_fuse_model.encode_text_m(
                input_ids=input_ids,
                input_segment_ids=input_segment_ids,
                input_mask=input_mask,
            )
            t_emb_m = t_output_m["pooled_output"]
            t_emb_m = self.text_visual_and_fuse_model.t_projector_m(t_emb_m)
            text_feat_m = F.normalize(t_emb_m, dim=-1)

            # t_tokens_m = t_output_m['encoded_layers'][-1]
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp

            sim_targets = torch.zeros(sim_i2t_m.size()).to(frames.device)
            sim_targets.fill_diagonal_(1)

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()

        # remove self-training persudo label(2023.01.03)
        # loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1),dim=1).mean()
        # loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1),dim=1).mean()

        loss_ita = (loss_i2t + loss_t2i) / 2.0

        self._dequeue_and_enqueue(image_feat_m, text_feat_m)

        # forward the positve video-text pair
        mmout_pos = self.text_visual_and_fuse_model.fusemodel(
            input_embs=t_tokens,
            input_segment_ids=input_segment_ids,
            input_mask=input_mask,
            frames_mask=frames_mask,
            visual_embeds=v_tokens,
        )
        cls_emb = mmout_pos["pooled_output"]

        with torch.no_grad():
            bs = frames.size(0)
            weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1) + 1e-4  # https://github.com/salesforce/BLIP/issues/76
            weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1) + 1e-4

            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        # select a negative image for each text
        image_embeds_neg = []
        frames_mask_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(v_tokens[neg_idx])
            frames_mask_neg.append(frames_mask[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
        frames_mask_neg = torch.stack(frames_mask_neg, dim=0)

        # select a negative text for each image
        text_embeds_neg = []
        text_segment_ids_neg = []
        text_mask_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(t_tokens[neg_idx])
            text_segment_ids_neg.append(input_segment_ids[neg_idx])
            text_mask_neg.append(input_mask[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_segment_ids_neg = torch.stack(text_segment_ids_neg, dim=0)
        text_mask_neg = torch.stack(text_mask_neg, dim=0)

        text_embeds_all = torch.cat([t_tokens, text_embeds_neg], dim=0)
        text_mask_all = torch.cat([input_mask, text_mask_neg], dim=0)
        text_segment_ids_all = torch.cat([input_segment_ids, text_segment_ids_neg], dim=0)

        image_embeds_all = torch.cat([image_embeds_neg, v_tokens], dim=0)
        frame_mask_all = torch.cat([frames_mask_neg, frames_mask], dim=0)

        # forward the negtive video-text pair
        mmout_neg = self.text_visual_and_fuse_model.fusemodel(
            input_embs=text_embeds_all,
            input_segment_ids=text_segment_ids_all,
            input_mask=text_mask_all,
            frames_mask=frame_mask_all,
            visual_embeds=image_embeds_all,
        )

        vl_embeddings = torch.cat([mmout_pos["pooled_output"], mmout_neg["pooled_output"]], dim=0)
        vl_output = self.classifier(vl_embeddings)

        itm_labels = torch.cat(
            [
                torch.ones(bs, dtype=torch.long),
                torch.zeros(2 * bs, dtype=torch.long),
            ],
            dim=0,
        ).to(frames.device)
        loss_itm = self.criterion(vl_output, itm_labels)

        cls_logits = self.category_classifier(cls_emb)
        train_rep_dict = {"logits": cls_logits, "labels": batch["labels"]}
        cls_loss = self.cal_cls_loss(**train_rep_dict)

        total_loss = loss_ita + loss_itm + cls_loss

        train_loss_dic = {
            "loss": total_loss,
            "cls_loss": cls_loss,
            "matching loss": loss_itm,
            "contrastive loss": loss_ita,
        }
        self.log_dict(train_loss_dic, console=True)
        return train_loss_dic

    def validation_step(self, batch, idx):
        input_ids, input_segment_ids, input_mask, frames, frames_mask = (
            batch["input_ids"],
            batch["input_segment_ids"],
            batch["input_mask"],
            batch["frames"],
            batch["frames_mask"],
        )

        t_output = self.text_visual_and_fuse_model.encode_text(
            input_ids=input_ids,
            input_segment_ids=input_segment_ids,
            input_mask=input_mask,
        )
        t_emb = t_output["pooled_output"]
        t_emb = self.text_visual_and_fuse_model.t_projector(t_emb)
        text_feat = F.normalize(t_emb, dim=-1)

        t_tokens = t_output["encoded_layers"][-1]

        v_output = self.text_visual_and_fuse_model.encode_image(images=frames, images_mask=frames_mask)
        v_emb = v_output["pooled_output"]
        v_emb = self.text_visual_and_fuse_model.v_projector(v_emb)
        image_feat = F.normalize(v_emb, dim=-1)

        v_tokens = v_output["encoded_layers"][-1][:, 1:, :]

        mmout = self.text_visual_and_fuse_model.fusemodel(
            input_embs=t_tokens,
            input_segment_ids=input_segment_ids,
            input_mask=input_mask,
            frames_mask=frames_mask,
            visual_embeds=v_tokens,
        )
        cls_emb = mmout["pooled_output"]
        cls_logits = self.category_classifier(cls_emb)
        val_rep_dic = {"logits": cls_logits, "labels": batch["labels"]}
        val_loss = self.cal_cls_loss(**val_rep_dict)
        val_dic = {"val_loss": val_loss}
        val_dic.update(val_rep_dic)
        return val_dic

    @torch.no_grad()
    def validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        gathered_results = DIST_ENV.all_gather_object(outputs)
        all_results = []
        for item in gathered_results:
            all_results.extend(item)
        all_logits = []
        all_labels = []

        for out in all_results:
            all_logits.extend(out["logits"].detach().cpu().tolist())
            all_labels.extend(out["labels"].detach().cpu().tolist())

        logits = torch.from_numpy(np.array(all_logits))
        labels = torch.from_numpy(np.array(all_labels))

        accuracy = compute_accuracy(logits, labels)

        val_metric_dict = {"val_acc": accuracy}
        self.log_dict(val_metric_dict, console=True)

    def cal_cls_loss(self, **kwargs):
        for key in ["logits", "labels"]:
            if key in kwargs:
                kwargs[key] = self.all_gather(kwargs[key].contiguous())
                kwargs[key] = kwargs[key].flatten(0, 1)
        loss = self.criterion(kwargs["logits"], kwargs["labels"])
        return loss

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
            elif n.startswith("text_visual_and_fuse_model"):
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

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr : ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr : ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1.0 - self.momentum)

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # world_size = int(os.environ.get('WORLD_SIZE') or 1)
    # tensors_gather = [torch.ones_like(tensor) for _ in range(world_size)]
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


if __name__ == "__main__":
    cli = CruiseCLI(
        MMClsModel,
        trainer_class=CruiseTrainer,
        datamodule_class=ByteDriveDataModule,
        trainer_defaults={
            "log_every_n_steps": 20,
            "max_epochs": 30,
            "enable_versions": True,
            "val_check_interval": 50.0,
            "sync_batchnorm": True,
            "find_unused_parameters": True,
            "summarize_model_depth": 5,
            "checkpoint_monitor": "loss",
            "checkpoint_mode": "min",
            "default_root_dir": "/mnt/bn/multimodel-pretrain/scripts/cruise_logs/pretrain_test",
        },
    )
    cfg, trainer, model, datamodule = cli.parse_args()
    print_cfg(cfg)
    trainer.fit(model, datamodule)
