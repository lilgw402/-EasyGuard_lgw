# -*- coding:utf-8 -*-
# Email:    jiangxubin@bytedance.com
# Created:  2023-03-02 14:23:40
# Modified: 2023-03-02 14:23:40
from typing import Optional

import torch
import torch.nn as nn
from models.modules.encoders import AutoDisBucketEncoder
from models.modules.losses import BCEWithLogitsLoss
from models.modules.mtl import AitmMtl
from models.modules.running_metrics import GeneralClsMetric
from models.template_models.MtlGandalfCruiseModule import MtlGandalfCruiseModule
from utils.registry import MODELS
from utils.util import count_params

from easyguard.core import AutoModel


@MODELS.register_module()
class EcomLiveGandalfAutoDisNNAsrAitmCruiseModel(MtlGandalfCruiseModule):
    def __init__(self, features, asr_encoder, mtl, embedding, kwargs, type=None):
        super(EcomLiveGandalfAutoDisNNAsrAitmCruiseModel, self).__init__(kwargs)
        self.save_hparams()

    def setup(self, stage: Optional[str] = None) -> None:
        # dense feature config
        self._feature_num = self.hparams.features.get("feature_num", 150)
        self._feature_input_num = self._feature_num - len(self.hparams.features.get("slot_mask", []))
        self._bucket_num = self.hparams.features.get("bucket_num", 8)
        self._bucket_dim = self.hparams.features.get("bucket_dim", 128)
        self._bucket_output_size = self.hparams.features.get("bucket_output_size", 1024)
        self._bucket_all_emb_dim = self._feature_input_num * self._bucket_dim
        # Text model config
        self._asr_embedding_size = self.hparams.kwargs.get("asr_embedding_size", 768)
        self._enable_asr_embedding = self.hparams.kwargs.get("enable_asr_embedding", 0)
        self._drop_prob = self.hparams.kwargs.get("dropout", 0.3)
        # Init loss weight
        self._loss_weight = self.hparams.mtl.get("loss_weight", None)
        self._constraint_weight = self.hparams.mtl.get("constraint_weight", 0.3)
        # Reset params
        if self.hparams.kwargs.get("reset_params", False):
            self._reset_params()
        # Init encoders
        self._init_encoders()
        # Init metric
        self._metric = GeneralClsMetric()
        self._train_logging_names = ["censor_label", "reject_label"]
        self._eval_output_names = ["loss", "censor_loss", "reject_loss"]
        self._metric_params["censor"] = {
            "score_key": "censor_prob",
            "label_key": "censor_label",
            "type": "ClsMetric",
        }
        self._metric_params["reject"] = {
            "score_key": "reject_prob",
            "label_key": "reject_label",
            "type": "ClsMetric",
        }
        self._parse_eval_output_advanced_metrics()
        print("_all_gather_output_names", self._all_gather_output_names)
        print("_eval_output_names", self._eval_output_names)
        count_params(self)

    def forward(
        self,
        auto_dis_input,
        feature_dense,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        censor_label=None,
        reject_label=None,
    ):
        # get auto_dis embedding
        auto_dis_embedding = self._auto_dis_bucket_encoder(auto_dis_input)
        # get concat features
        asr_output = self._asr_encoder(input_ids, attention_mask, token_type_ids, output_pooled=True)
        asr_embedding = asr_output["pooled_output"]
        # Add dropout for embedding
        asr_embedding = self._asr_emb_dropout(asr_embedding)
        # Concat all input features which have been transformed into embeddings
        concat_features = self._bucket_embedding_bottom(
            torch.cat([auto_dis_embedding, feature_dense, asr_embedding], 1)
        )
        # Get output of model
        censor_output, reject_output = self._aitm_mtl(concat_features)
        # output = self._classifier_final(concat_features)
        if censor_label is not None and reject_label is not None:
            loss_dict = self._create_loss(
                censor_output,
                reject_output,
                censor_label,
                reject_label,
                constraint_weight=self._constraint_weight,
                loss_weight=self._loss_weight,
            )
            censor_prob = self._post_process_output(censor_output)
            reject_prob = self._post_process_output(reject_output)
            output_dict = {
                "censor_prob": censor_prob,
                "reject_prob": reject_prob,
            }
            # 添加评价指标
            censor_eval_dict = self._metric.batch_eval(censor_prob, censor_label, key="censor")
            reject_eval_dict = self._metric.batch_eval(reject_prob, reject_label, key="reject")
            output_dict.update(censor_eval_dict)
            output_dict.update(reject_eval_dict)
            output_dict.update(loss_dict)
            return loss_dict, output_dict
        else:
            return self._post_process_output(censor_output), self._post_process_output(reject_output)

    def pre_process_inputs(self, batched_feature_data):
        batched_feature_data_items = [
            batched_feature_data["auto_dis_input"],
            batched_feature_data["feature_dense"],
            batched_feature_data["input_ids"],
            batched_feature_data["attention_mask"],
            batched_feature_data["token_type_ids"],
        ]
        return self._pre_process(batched_feature_data_items)

    def pre_process_targets(self, batched_feature_data):
        batched_feature_data_items = [
            batched_feature_data["censor_label"],
            batched_feature_data["reject_label"],
        ]
        return self._pre_process(batched_feature_data_items)

    def trace_before_step(self, batch):
        # tracer don't like dict of input
        return self.pre_process_inputs(batch)

    def trace_step(self, batch):
        trace_output = self.forward(*batch)
        return trace_output

    def trace_after_step(self, result):
        # 按照本文档导出无需实现该方法，留空即可
        pass

    def _init_encoders(self):
        # Init bucket encoder
        self._auto_dis_bucket_encoder = AutoDisBucketEncoder(
            feature_num=self._feature_input_num,
            bucket_num=self._bucket_num,
            bucket_dim=self._bucket_dim,
            output_size=self._bucket_output_size,
            use_fc=False,
            add_block=False,
        )
        # Init model components:asr
        self._asr_encoder_param = self.hparams.asr_encoder
        self._asr_encoder = AutoModel.from_pretrained(
            self._asr_encoder_param.get("encoder_name", "fashion-deberta-asr-small"),
            n_layers=self._asr_encoder_param.get("num_hidden_layers", 3),
        )
        self._asr_emb_dropout = self._init_emb_dropout()
        # Init multi-task components:aitm
        self._aitm_mtl = AitmMtl(input_size=512, shared_out_size=256, tower_dims=[256, 128, 64])
        self._init_cls_layer()
        self._init_criterion()

    def _init_nn_bottom(self, dim: list):  # net
        net = nn.Sequential()
        for i in range(len(dim) - 1):
            net.add_module("layer_{}".format(i), nn.Linear(dim[i], dim[i + 1], bias=True))
            net.add_module("activation_{}".format(i), nn.ReLU(inplace=True))
            net.add_module(
                "dropout_{}".format(i),
                nn.Dropout(p=self._drop_prob, inplace=False),
            )
        return net

    def _init_emb_dropout(self):
        hidden_dropout_prob = self._asr_encoder_param.get("emb_dropout_prob", 0)
        emb_dropout = 0 if not hidden_dropout_prob else hidden_dropout_prob
        return nn.Dropout(emb_dropout)

    def _init_cls_layer(self):
        # bottom
        self._bucket_embedding_bottom = self._init_nn_bottom(
            dim=[
                self._bucket_all_emb_dim + self._feature_input_num + self._asr_encoder_param.get("embedding_dim", 768),
                1024,
                512,
            ]
        )
        # original classifier for single task reject
        self._classifier_final = nn.Sequential(
            nn.Linear(512, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self._drop_prob, inplace=False),
            nn.Linear(256, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self._drop_prob, inplace=False),
            nn.Linear(128, 1, bias=True),
        )

    def _init_criterion(self):
        self._censor_criterion_final = BCEWithLogitsLoss()
        self._reject_criterion_final = BCEWithLogitsLoss()

    def _create_loss(
        self,
        censor_output,
        reject_output,
        censor_label,
        reject_label,
        constraint_weight=0.3,
        loss_weight=None,
    ):
        censor_loss = self._censor_criterion_final(censor_output, censor_label)
        reject_loss = self._reject_criterion_final(reject_output, reject_label)

        label_constraint = torch.maximum(
            self._post_process_output(reject_output) - self._post_process_output(censor_output),
            torch.zeros_like(reject_label),
        )
        constraint_loss = torch.sum(label_constraint)
        if not loss_weight:
            loss = censor_loss + reject_loss + constraint_weight * constraint_loss
        else:
            loss = loss_weight[0] * censor_loss + loss_weight[1] * reject_loss + constraint_weight * constraint_loss
        return {
            "loss": loss,
            "censor_loss": censor_loss,
            "reject_loss": reject_loss,
        }
