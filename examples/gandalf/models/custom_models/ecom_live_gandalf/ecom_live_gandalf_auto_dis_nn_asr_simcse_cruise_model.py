# -*- coding:utf-8 -*-
# Email:    jiangxubin@bytedance.com
# Created:  2023-03-02 14:23:40
# Modified: 2023-03-02 14:23:40
# coding=utf-8
# Author: jiangxubin
# Create: 2022/8/2 18:15
from typing import Optional

import torch
import torch.nn as nn
from models.modules.encoders import AutoDisBucketEncoder
from models.modules.losses import BCEWithLogitsLoss
from models.modules.running_metrics import GeneralClsMetric
from models.template_models.GandalfCruiseModule import GandalfCruiseModule
from transformers import AutoModel
from utils.registry import MODELS
from utils.util import count_params


@MODELS.register_module()
class EcomLiveGandalfAutoDisNNAsrSimcseCruiseModel(GandalfCruiseModule):
    def __init__(self, features, asr_encoder, embedding, kwargs, type=None):
        super(EcomLiveGandalfAutoDisNNAsrSimcseCruiseModel, self).__init__(kwargs)
        self.save_hparams()

    def setup(self, stage: Optional[str] = None) -> None:
        # print(self.hparams)
        self._feature_num = self.hparams.features.get("feature_num", 150)
        self._feature_input_num = self._feature_num - len(self.hparams.features.get("slot_mask", []))
        self._bucket_num = self.hparams.features.get("bucket_num", 8)
        self._bucket_dim = self.hparams.features.get("bucket_dim", 128)
        self._bucket_output_size = self.hparams.features.get("bucket_output_size", 1024)
        # Advanced model config
        self._enable_asr_embedding = self.hparams.kwargs.get("enable_asr_embedding", 0)
        self._drop_prob = self.hparams.kwargs.get("dropout", 0.3)
        self._bucket_all_emb_dim = self._feature_input_num * self._bucket_dim
        # Init loss weight
        self._loss_weight = self.hparams.kwargs.get("loss_weight", None)
        # Reset params
        if self.hparams.kwargs.get("reset_params", False):
            self._reset_params()
        # Init encoders
        self.init_encoders()
        # Init metric
        self._metric = GeneralClsMetric()
        count_params(self)

    def forward(
        self,
        auto_dis_input,
        feature_dense,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        targets=None,
    ):
        # get auto_dis embedding
        auto_dis_embedding = self._auto_dis_bucket_encoder(auto_dis_input)
        # get concat features
        asr_output = self._asr_encoder(input_ids, attention_mask, token_type_ids)
        asr_embedding = asr_output["pooler_output"]
        # Add dropout for embedding
        asr_embedding = self._asr_emb_dropout(asr_embedding)
        # Concat all input features which have been transformed into embeddings
        concat_features = self._bucket_embedding_bottom(
            torch.cat([auto_dis_embedding, feature_dense, asr_embedding], 1)
        )
        # Get output of model
        output = self._classifier_final(concat_features)
        if targets is not None:
            if not self._loss_weight:
                loss = self._criterion_final(output, targets)
                loss_dict = {"loss": loss}
                output_prob = self._post_process_output(output)
                output_dict = {"output": output_prob}
            else:
                loss = self._criterion_final(output, targets)
                loss_dict = {
                    "loss": loss,
                }
                output_prob = self._post_process_output(output)
                output_dict = {
                    "output": output_prob,
                }
            # 添加评价指标
            eval_dict = self._metric.batch_eval(output_prob, targets)
            output_dict.update(eval_dict)
            output_dict.update(loss_dict)
            return loss_dict, output_dict
        else:
            return self._post_process_output(output)

    def pre_process_inputs(self, batched_feature_data):
        batched_feature_data_items = [
            batched_feature_data["auto_dis_input"],
            batched_feature_data["feature_dense"],
            batched_feature_data["input_ids"],
            batched_feature_data["attention_mask"],
            batched_feature_data["token_type_ids"],
        ]
        return self._pre_process(batched_feature_data_items)

    def init_encoders(self):
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
            "/mlx_devbox/users/jiangxubin/repo/EasyGuard/examples/gandalf//models/weights/simcse_bert_base",
            num_hidden_layers=3,
            num_attention_heads=6,
        )
        self._asr_emb_dropout = self._init_emb_dropout()
        self._init_cls_layer()
        self._init_criterion()

    def _nn_bottom(self, dim: list):  # net
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
        self._bucket_embedding_bottom = self._nn_bottom(
            dim=[
                self._bucket_all_emb_dim + self._feature_input_num + self._asr_encoder_param.get("embedding_dim", 768),
                1024,
                512,
            ]
        )
        # original classifier for pass/ban
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
        self._criterion_final = BCEWithLogitsLoss()

    def trace_before_step(self, batch):
        # tracer don't like dict of input
        return self.pre_process_inputs(batch)

    def trace_step(self, batch):
        (
            auto_dis_input,
            feature_dense,
            input_ids,
            attention_mask,
            token_type_ids,
        ) = batch
        trace_output = self.forward(
            auto_dis_input,
            feature_dense,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return trace_output
