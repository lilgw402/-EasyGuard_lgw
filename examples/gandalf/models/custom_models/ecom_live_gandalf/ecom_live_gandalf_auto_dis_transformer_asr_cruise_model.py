# coding=utf-8
# Author: jiangxubin@bytedance.com
# Create: 2022/08/08 18:41
from typing import Optional

import torch
import torch.nn as nn
from models.modules.encoders import AutoDisBucketEncoder
from models.modules.encoders.Transformer import TransformerEncoderv2
from models.modules.losses import BCEWithLogitsLoss
from models.modules.running_metrics import GeneralClsMetric
from models.template_models.GandalfCruiseModule import GandalfCruiseModule
from utils.registry import MODELS
from utils.util import count_params

from easyguard.core import AutoModel


@MODELS.register_module()
class EcomLiveGandalfAutoDisTransformerAsrCruiseModel(GandalfCruiseModule):
    def __init__(self, features, asr_encoder, embedding, kwargs, type=None):
        super(EcomLiveGandalfAutoDisTransformerAsrCruiseModel, self).__init__()
        self.save_hparams()

    def setup(self, stage: Optional[str] = None) -> None:
        # Dense feature config
        self._feature_num = self.hparams.features.get("feature_num", 150)
        self._feature_input_num = self._feature_num - len(self.hparams.features.get("slot_mask", []))
        self._bucket_num = self.hparams.features.get("bucket_num", 8)
        self._bucket_dim = self.hparams.features.get("bucket_dim", 128)
        self._bucket_output_size = self.hparams.features.get("bucket_output_size", 1024)
        # Transformer config
        self._fusion_transformer_num_heads = self.hparams.kwargs.get("num_heads", 6)
        self._fusion_transformer_num_layers = self.hparams.kwargs.get("num_layers", 3)
        # Advanced model config
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
        # Get AutoDis embedding
        auto_dis_embedding = self._auto_dis_bucket_encoder(auto_dis_input)
        auto_dis_embedding_seq = auto_dis_embedding.reshape(-1, self._feature_num, self._bucket_dim)
        # Get feature dense emd
        feature_dense_emb = self._feature_dense_proj(feature_dense).unsqueeze(1)
        # transformer结构
        asr_embedding = self._asr_encoder(input_ids, attention_mask, token_type_ids, output_pooled=True)
        proj_asr_embedding = self._asr_embedding_proj(asr_embedding).unsqueeze(1)
        embeddings = torch.cat(
            [
                auto_dis_embedding_seq,
                feature_dense_emb,
                proj_asr_embedding,
            ],
            dim=1,
        )
        # features
        embeddings = embeddings.transpose(0, 1)  # embeddings.permute(1, 0, 2)
        output_seq = self._transformer_encoder(embeddings)  # output_seq, layer_output_seq
        # output
        output = self._classifier_final(output_seq[0])

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
            self._asr_encoder_param.get("encoder_name", "fashion-deberta-asr-small"),
            n_layers=self._asr_encoder_param.get("num_hidden_layers", 3),
        )
        self._transformer_encoder = TransformerEncoderv2(
            embed_dim=self._bucket_dim,
            num_heads=self._fusion_transformer_num_heads,
            layers=self._fusion_transformer_num_layers,
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

    @property
    def classifier_final(self):
        return self._classifier_final

    def _init_project_layers(self):
        self._feature_dense_proj = nn.Linear(self._feature_num, self._bucket_dim)
        self._asr_embedding_proj = nn.Linear(self._asr_embedding_dim, self._bucket_dim)
