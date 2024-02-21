# -*- coding: utf-8 -*-
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
from transformers import BertModel

from easyguard.modelzoo.models.albert import ALBert
from easyguard.modelzoo.models.swin import SwinTransformer

from ...utils.losses import ArcMarginProduct, LearnableNTXentLoss, LearnablePCLLoss, SCELoss, cross_entropy
from .convnext import ConvNeXt
from .roberta import RoBerta
from .temporal_fusion import TemporalFusion
from .utils import CosineAnnealingWarmupRestarts, accuracy


def p_fix_r(output, labels, fix_r):
    output_sort = output[(-output).argsort()]
    labels_sort = labels[(-output).argsort()]
    num_pos = np.sum(labels == 1)
    recall_sort = np.cumsum(labels_sort) / float(num_pos)
    index = np.abs(recall_sort - fix_r).argmin()
    thr = output_sort[index]
    precision = np.sum(((output >= thr) == labels) * labels) / np.sum(output >= thr)
    return precision, recall_sort[index], thr


class AuthenticMM(CruiseModule):
    def __init__(
        self,
        config_text: str = "/mnt/bn/ecom-govern-maxiangqian/qingxuan/EasyGuard/examples/high_quality_live/config/config_text.yaml",
        config_visual: str = "/mnt/bn/ecom-govern-maxiangqian/qingxuan/EasyGuard/examples/high_quality_live/config/config_visual.yaml",
        config_fusion: str = "/mnt/bn/ecom-govern-maxiangqian/qingxuan/EasyGuard/examples/high_quality_live/config/config_fusion.yaml",
        learning_rate: float = 1e-4,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.02,
        lr_warmup_steps_weight: float = 0,
        load_pretrained: str = None,
        use_text_modal: bool = True,
        use_visual_modal: bool = True,
        num_lvl2_labels: int = 12,
        use_multilabel: bool = False,
        use_arcface: bool = False,
    ):
        super(AuthenticMM, self).__init__()
        self.save_hparams()

    def setup(self, stage) -> None:
        """
        Load yaml file as config class
        """
        with open(self.hparams.config_text) as fp:
            self.config_text = SimpleNamespace(**yaml.load(fp, yaml.Loader))
        with open(self.hparams.config_visual) as fp:
            self.config_visual = SimpleNamespace(**yaml.load(fp, yaml.Loader))
        with open(self.hparams.config_fusion) as fp:
            self.config_fusion = SimpleNamespace(**yaml.load(fp, yaml.Loader))

        """
        Initialize modules
        """
        if self.config_text.name == "albert":
            self.text = ALBert(self.config_text)
        elif self.config_text.name == "roberta":
            self.text = RoBerta(
                transformer=BertModel,
                bert_dir=self.config_text.bert_dir,
                mlm_enable=self.config_text.mlm_enable,
                embedder_only=self.config_text.embedder_only,
                with_hidden_states=self.config_text.with_hidden_states,
                out_channels=self.config_text.out_channels,
            )
        if self.config_visual.name == "swin":
            self.visual = SwinTransformer(
                img_size=self.config_visual.img_size,
                num_classes=self.config_visual.output_dim,
                embed_dim=self.config_visual.embed_dim,
                depths=self.config_visual.depths,
                num_heads=self.config_visual.num_heads,
            )
        elif self.config_visual.name == "convnext":
            self.visual = ConvNeXt(depths=self.config_visual.depths, dims=self.config_visual.dims)

        if self.config_visual.pooling_type:
            self.frame_fusion = TemporalFusion(
                self.config_visual.pooling_type,
                self.config_visual.output_dim,
                self.config_visual.num_frames,
            )

        self.t_projector = nn.Linear(self.config_text.hidden_size, self.config_fusion.hidden_size)
        self.v_projector = nn.Linear(self.config_visual.output_dim, self.config_fusion.hidden_size)

        if self.config_fusion.name == "transformer":
            self.cls_emb = nn.Embedding(1, self.config_fusion.hidden_size)
            self.ln_text = nn.LayerNorm(self.config_fusion.hidden_size)
            self.ln_visual = nn.LayerNorm(self.config_fusion.hidden_size)
            self.fuse = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=self.config_fusion.hidden_size,
                    nhead=8,
                    batch_first=True,
                    activation="gelu",
                ),
                num_layers=self.config_fusion.num_layers,
            )
            self.fuse_dropout = nn.Dropout(0.2)

        """
        Initialize output layer
        """
        if self.config_fusion.name == "transformer":
            fuse_emb_size = self.config_fusion.hidden_size
        elif self.config_fusion.name == "concat":
            if self.config_visual.pooling_type:
                fuse_emb_size = self.config_fusion.hidden_size * 3
            else:
                fuse_emb_size = self.config_fusion.hidden_size * (2 + self.config_visual.num_frames)
        self.fuse_category_lvl1 = nn.Sequential(
            nn.Linear(fuse_emb_size, self.config_fusion.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(self.config_fusion.embd_pdrop),
            nn.Linear(
                self.config_fusion.hidden_size * 2,
                self.config_fusion.category_logits_level1,
            ),
        )
        if self.hparams.use_multilabel:
            self.fuse_category_lvl2 = nn.Sequential(
                nn.Linear(fuse_emb_size, self.config_fusion.hidden_size * 2),
                nn.GELU(),
                nn.Dropout(self.config_fusion.embd_pdrop),
                nn.Linear(
                    self.config_fusion.hidden_size * 2,
                    self.hparams.num_lvl2_labels,
                ),
            )
        self.softmax = nn.Softmax(dim=1)

        """
        Initialize loss
        """
        if self.hparams.use_arcface:
            self.arc = ArcMarginProduct(
                in_features=fuse_emb_size,
                out_features=self.config_fusion.category_logits_level1,
            )

        """
        Initialize some fixed parameters.
        """
        # # HARD CODE!!!
        # self.PAD = 2
        # self.MASK = 1
        # self.SEP = 3
        # # 文本的position ids
        # self.text_position_ids = nn.Parameter(
        #     data=torch.arange(0, 512, dtype=torch.int64),
        #     requires_grad=False
        # )  # [512, ]
        # self.text_segment_ids = nn.Parameter(
        #     data=torch.zeros(size=(512,), dtype=torch.int64),
        #     requires_grad=False
        # )  # [512, ]

        self.initialize_weights()

    def initialize_weights(self):
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
            if self.config_text.name == "albert":
                # load weights of text backbone
                text_pretrained = (
                    "hdfs://haruna/home/byte_search_nlp_lq/multimodal/modelhub/videoclip_swin_dy_20211206/model.th"
                )
                text_backbone = load(text_pretrained, map_location="cpu")
                state_dict_new = OrderedDict()
                for key, value in text_backbone.items():
                    if key.startswith("falbert"):
                        trimmed_key = key[len("falbert.") :]
                    else:
                        trimmed_key = key
                    if trimmed_key.split(".")[0] in ["encoder", "embedding"]:
                        state_dict_new[trimmed_key] = value
                missing_keys, unexpected_keys = self.text.load_state_dict(state_dict_new, strict=False)
                print("albert pretrained load! {} keys".format(len(state_dict_new.keys())))
                print("missing_keys: ", missing_keys)
                print("unexpected_keys: ", unexpected_keys)
            if self.config_visual.name == "swin":
                # load weights of vitual backbone
                visual_pretrained = (
                    "hdfs://haruna/home/byte_search_nlp_lq/multimodal/modelhub/videoclip_swin_dy_20211206/model.th"
                )
                visual_backbone = load(visual_pretrained, map_location="cpu")
                state_dict_new = OrderedDict()
                for key, value in visual_backbone.items():
                    if key.startswith("falbert"):
                        trimmed_key = key[len("falbert.") :]
                    else:
                        trimmed_key = key
                    if trimmed_key[:6] == "visual" and trimmed_key:
                        state_dict_new[trimmed_key[7:]] = value
                missing_keys, unexpected_keys = self.visual.load_state_dict(state_dict_new, strict=False)
                print("swin pretrained load! {} keys".format(len(state_dict_new.keys())))
                print("missing_keys: ", missing_keys)
                print("unexpected_keys: ", unexpected_keys)
            elif self.config_visual.name == "convnext":
                convnext_checkpoint = torch.load(self.config_visual.pretrained, map_location="cpu")
                del convnext_checkpoint["model"]["head.weight"]
                del convnext_checkpoint["model"]["head.bias"]
                self.visual.load_state_dict(convnext_checkpoint["model"])

        # if hexists(self.hparams.load_pretrained):
        #     state_dict_ori = self.state_dict()
        #     state_dict = load(self.hparams.load_pretrained, map_location="cpu")
        #     state_dict_new = OrderedDict()
        #     for key, value in state_dict.items():
        #         if key in state_dict_ori and state_dict_ori[key].shape == state_dict[key].shape:
        #             state_dict_new[key] = value
        #     print("load from pretrained model, {} key matched.".format(len(state_dict_new)))

    def forward_text(self, token_ids: torch.Tensor, attn_mask=None, segment_ids=None):
        # token_ids: [B, S, T]
        B, S, T = token_ids.shape
        token_ids = token_ids.view(-1, T)
        attn_mask = attn_mask.view(-1, T)
        segment_ids = segment_ids.view(-1, T)
        if self.config_text.name == "roberta":
            t_rep = self.text(token_ids, attn_mask, segment_ids)
        elif self.config_text.name == "albert":
            t_out = self.text(
                input_ids=token_ids,
                input_segment_ids=segment_ids,
                input_mask=attn_mask,
            )
            t_rep = t_out["pooled_output"]
        # text_masks = (token_ids != self.PAD).long()
        # text_segment_ids = (token_ids == self.SEP).long()
        # text_segment_ids = torch.cumsum(text_segment_ids, dim=-1) - text_segment_ids
        # text_segment_ids = torch.clamp(text_segment_ids, min=0, max=1)
        # batch_size, text_length = token_ids.shape
        # position_ids = self.text_position_ids[:text_length].unsqueeze(0).expand((batch_size, -1))  # [B, M]

        # t_out = self.text(input_ids=token_ids,
        #                   input_segment_ids=text_segment_ids,
        #                   input_mask=text_masks,
        #                   position_ids=position_ids)
        # t_rep = t_out['pooled_output']  # img_out: [B * S, d_t]
        t_rep = self.t_projector(t_rep)  # img_out: [B * S, d_f]
        t_rep = t_rep.view(B, S, -1)
        return t_rep

    def forward_image(self, image: torch.Tensor):
        # image: [B, S, C, H, W]
        B, S, C, H, W = image.shape
        image = image.view(-1, C, H, W)
        img_out = self.visual(image)  # img_out: [B * S, d_v]
        # v_rep = self.v_projector(img_out)  # v_rep: [B * S, d_f]
        v_rep = img_out.view(B, S, -1)
        if self.config_visual.pooling_type:
            v_rep = self.frame_fusion(v_rep)
            v_rep = v_rep.view(B, 1, -1)
        v_rep = self.v_projector(v_rep)
        return v_rep

    def forward_fuse(self, t_emb=None, v_emb=None):
        # t_emb: [B, S_t, d_f]
        # v_emb: [B, S_v, d_f]
        # generate cls emb
        if self.config_fusion.name == "transformer":
            batch_size = v_emb.shape[0] if v_emb is not None else t_emb.shape[0]
            device = v_emb.device if v_emb is not None else t_emb.device
            cls_emb = self.cls_emb(torch.zeros(batch_size, device=device).long()).unsqueeze(1)  # [B, 1, d_f]
            if t_emb is not None and v_emb is not None:
                fuse_emb = self.fuse_dropout(
                    torch.cat(
                        [cls_emb, self.ln_visual(v_emb), self.ln_text(t_emb)],
                        dim=1,
                    )
                )  # [B, 1 + S_v + S_t, d_f]
            elif t_emb is not None:
                # 文本单模态
                fuse_emb = self.fuse_dropout(torch.cat([cls_emb, self.ln_text(t_emb)], dim=1))  # [B, 1 + S_t, d_f]
            else:
                # 视觉单模态
                fuse_emb = self.fuse_dropout(torch.cat([cls_emb, self.ln_visual(v_emb)], dim=1))  # [B, 1 + S_v, d_f]
            fuse_emb = self.fuse(fuse_emb)
            fuse_emb = fuse_emb[:, 0]
        elif self.config_fusion.name == "concat":
            batch_size = v_emb.shape[0] if v_emb is not None else t_emb.shape[0]
            if t_emb is not None and v_emb is not None:
                v_emb = v_emb.view(batch_size, -1)
                t_emb = t_emb.view(batch_size, -1)
                fuse_emb = torch.cat([v_emb, t_emb], dim=1)
            elif t_emb is not None:
                t_emb = t_emb.view(batch_size, -1)
                fuse_emb = t_emb
            else:
                v_emb = v_emb.view(batch_size, -1)
                fuse_emb = v_emb
        return fuse_emb

    def forward(
        self,
        token_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        attn_mask: torch.Tensor,
        image: torch.Tensor,
        label: torch.Tensor = None,
        **kwargs,
    ):
        t_emb, v_emb = None, None
        if self.hparams.use_text_modal:
            t_emb = self.forward_text(token_ids, attn_mask, segment_ids)
        if self.hparams.use_visual_modal:
            v_emb = self.forward_image(image)
        fuse_emb = self.forward_fuse(t_emb, v_emb)
        if self.hparams.use_arcface:
            arc_logit = self.arc(fuse_emb, label)
        out_lvl1 = self.fuse_category_lvl1(fuse_emb)
        if self.hparams.use_arcface:
            ret = {
                "fuse_rep": fuse_emb,
                "out_lvl1": out_lvl1,
                "arc_out": arc_logit,
            }
        else:
            ret = {
                # "t_rep": t_emb,
                # "v_rep": v_emb,
                "fuse_rep": fuse_emb,
                "out_lvl1": out_lvl1,
            }

        if self.hparams.use_multilabel:
            out_lvl2 = self.fuse_category_lvl2(fuse_emb)
            ret.update(
                {
                    "out_lvl2": out_lvl2,
                }
            )

        return ret

    def cal_cls_loss(self, **kwargs):
        if self.hparams.use_arcface:
            key_list = ["out_lvl1", "arc_out", "label"]
        else:
            key_list = ["out_lvl1", "label"]
        for key in key_list:
            if key in kwargs:
                kwargs[key] = self.all_gather(kwargs[key].contiguous())
                kwargs[key] = kwargs[key].flatten(0, 1)
        loss = cross_entropy(kwargs["out_lvl1"], kwargs["label"])
        if self.hparams.use_arcface:
            arc_loss = cross_entropy(kwargs["arc_out"], kwargs["label"])
            total_loss = loss + 0.5 * arc_loss
            return {"ce_loss": total_loss}
        else:
            return {"ce_loss": loss}

    def cal_bce_loss(self, **kwargs):
        for key in ["out_lvl2", "leaf_label"]:
            if key in kwargs:
                kwargs[key] = self.all_gather(kwargs[key].contiguous())
                kwargs[key] = kwargs[key].flatten(0, 1)
        # HARD CODE!!!
        pos_weight = torch.Tensor(
            [
                1.0,
                0.3,
                0.3,
                0.3,
                1.0,
                1.0,
                1.0,
                0.1,
                0.1,
                1.0,
                1.0,
                1.0,
            ]
        ).to(kwargs["out_lvl2"].device)
        ###############
        loss = F.binary_cross_entropy_with_logits(
            kwargs["out_lvl2"],
            kwargs["leaf_label"].float(),
            pos_weight=pos_weight,
        )
        return {"bce_loss": loss}

    def merge_loss(self, losses, weights):
        loss_dict = {}
        final_loss = 0
        for i in range(len(losses)):
            for l_name, l_value in losses[i].items():
                loss_dict[l_name] = l_value
                final_loss += weights[i] * l_value
        loss_dict["loss"] = final_loss
        return loss_dict

    def training_step(self, batch, idx):
        token_ids, segment_ids, attn_mask, image, label = (
            batch["token_ids"],
            batch["segment_ids"],
            batch["attn_mask"],
            batch["frames"],
            batch["label"],
        )
        rep_dict = self.forward(token_ids, segment_ids, attn_mask, image, label)
        rep_dict.update(batch)

        loss_dict = self.cal_cls_loss(**rep_dict)
        if self.hparams.use_multilabel:
            bce_loss_dict = self.cal_bce_loss(**rep_dict)
            loss_dict = self.merge_loss([loss_dict, bce_loss_dict], [1.0, 1.0])
        else:
            loss_dict = self.merge_loss(
                [
                    loss_dict,
                ],
                [
                    1.0,
                ],
            )

        acc, _, _ = accuracy(rep_dict["out_lvl1"], rep_dict["label"])
        gt = torch.eq(rep_dict["label"], 2).int()
        pred_score = self.softmax(rep_dict["out_lvl1"])[:, 2]
        loss_dict.update({"acc": acc, "b_pred": pred_score, "b_gt": gt})
        loss_dict.update({"learning_rate": self.trainer.lr_scheduler_configs[0].scheduler.get_lr()[0]})

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
        token_ids, image = batch["token_ids"], batch["frames"]
        rep_dict = self.forward(token_ids, image)
        return {
            "pred": self.softmax(rep_dict["out_lvl1"]),
            "label": batch["label"],
            "item_id": batch["item_id"],
        }

    def configure_optimizers(self):
        no_decay = ["bias", "bn", "norm"]
        no_dacay_params_dict = {"params": [], "weight_decay": 0.0}
        normal_params_dict = {
            "params": [],
            "weight_decay": self.hparams.weight_decay,
        }

        for n, p in self.named_parameters():
            if any(nd in n for nd in no_decay):
                no_dacay_params_dict["params"].append(p)
            else:
                normal_params_dict["params"].append(p)
        optimizer_grouped_parameters = [
            no_dacay_params_dict,
            normal_params_dict,
        ]

        if self.config_visual.name == "swin":
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                betas=self.hparams.betas,
                eps=self.hparams.eps,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.config_visual.name == "convnext":
            optimizer = torch.optim.SGD(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                momentum=0.9,
                weight_decay=self.hparams.weight_decay,
            )

        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.trainer.total_steps,
            cycle_mult=1.0,
            max_lr=self.hparams.learning_rate,
            min_lr=0,
            warmup_steps=int(self.trainer.total_steps * self.hparams.lr_warmup_steps_weight),
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
