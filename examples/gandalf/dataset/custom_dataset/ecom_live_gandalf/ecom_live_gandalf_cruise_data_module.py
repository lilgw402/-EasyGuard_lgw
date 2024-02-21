# -*- coding:utf-8 -*-
# Email:    jiangxubin@bytedance.com
# Created:  2023-02-27 20:36:34
# Modified: 2023-02-27 20:36:34
import json
import pickle
import re
from typing import Optional

import numpy as np
import torch
from addict import Dict
from dataset.template_datasets.GandalfCruiseDataModule import (
    GandalfParquetCruiseDataModule,
    GandalfParquetFeatureProvider,
)
from dataset.transforms.text_transforms.DebertaTokenizer import DebertaTokenizer

# from easyguard.core import AutoTokenizer
from transformers import AutoTokenizer
from utils.driver import get_logger
from utils.registry import DATASETS, FEATURE_PROVIDERS


@FEATURE_PROVIDERS.register_module()
class EcomLiveGandalfParquetAutoDisFeatureProvider(GandalfParquetFeatureProvider):
    def __init__(
        self,
        feature_num,
        max_len=512,
        use_high_precision=False,
        filtered_tags=None,
        slot_mask=None,
        feature_norm_info=None,
        embedding_conf=None,
        tokenizer_path=None,
        save_extra=False,
        eval_mode=False,
        trace_mode=False,
        **kwargs,
    ):
        super(EcomLiveGandalfParquetAutoDisFeatureProvider, self).__init__()
        self._slot_mask = slot_mask
        self._feature_num = feature_num
        self._use_high_precision = use_high_precision
        self._filtered_tags = filtered_tags
        self._feature_norm_info = feature_norm_info
        self._embedding_conf = embedding_conf
        self._feature_input_num = feature_num - len(slot_mask)
        self._active_slot = [i for i in range(self._feature_num) if i not in self._slot_mask]
        # self._text_tokenizer = AutoTokenizer.from_pretrained(self.asr_model_name, return_tensors="pt", max_length=512)        # noqa: E501
        # self._text_tokenizer = DebertaTokenizer(f'{tokenizer_path}/vocab.txt',max_len=max_len)
        self._text_tokenizer = AutoTokenizer.from_pretrained("./models/weights/simcse_bert_base")
        self._save_extra = save_extra

    def process_feature_dense(self, features):
        # 加载预处理参数
        active_slot = torch.tensor(self._active_slot, dtype=torch.long).reshape(len(self._active_slot))
        compressed_min = [self._feature_norm_info.get(str(slot_id), [0, 1])[0] for slot_id in self._active_slot]
        compressed_max = [self._feature_norm_info.get(str(slot_id), [0, 1])[1] for slot_id in self._active_slot]
        compressed_min = torch.tensor(compressed_min, dtype=torch.float32).reshape(-1)
        compressed_max = torch.tensor(compressed_max, dtype=torch.float32).reshape(-1)
        compressed_range = compressed_max - compressed_min
        # 并行特征预处理
        features = torch.tensor(features, dtype=torch.float32)
        feature_dense = features[active_slot]
        feature_dense_norm = (feature_dense - compressed_min) / compressed_range
        feature_dense_norm[feature_dense == -1] = 0.0  # 特征为-1, norm值为0
        feature_dense_norm[feature_dense.isnan()] = 0.0  # 特征缺失, norm值为0
        feature_dense_norm = torch.clamp(feature_dense_norm, min=0.0, max=1.0)
        auto_dis_input_list = [
            feature_dense_norm,
            feature_dense_norm * feature_dense_norm,
            torch.sqrt(feature_dense_norm),
        ]
        auto_dis_input = torch.stack(auto_dis_input_list, dim=1)
        return auto_dis_input, feature_dense_norm

    def process_text(self, text):
        # 加载预处理参数
        # asr_inputs = self._text_tokenizer(text)
        asr_inputs = self._text_tokenizer(
            text, max_length=512, padding="max_length", truncation=True
        )  # , return_tensors="pt")
        asr_inputs["input_ids"] = torch.tensor(asr_inputs["input_ids"], dtype=torch.int32)
        asr_inputs["attention_mask"] = torch.tensor(asr_inputs["attention_mask"], dtype=torch.int32)
        asr_inputs["token_type_ids"] = torch.tensor(asr_inputs["token_type_ids"], dtype=torch.int32)
        return asr_inputs

    def process(self, data):
        feature_data = {}
        # 数据mask:是否mask高准数据流
        source = data.get("source", "origin")
        if source == "high_precision" and not self._use_high_precision:
            return None
        # 标签mask:是否mask特定标签数据
        if self._filtered_tags and data.get("verify_reason", ""):
            if len(re.findall(self._filtered_tags, data.get("verify_reason", ""))) > 0:
                # get_logger().info("Mask data by filtered tags:{}".format(data.get('verify_reason','')))
                return None
        # parquet数据主要包含labels,features,contents,embeddings字段，而且均已经序列化了
        labels = pickle.loads(data["labels"])
        # numerical features
        features = pickle.loads(data["features"])
        strategy = list(features["strategy"].values())
        # content features
        contents = pickle.loads(data["contents"])
        asr = contents["asr"]
        # embedding features
        embeddings = pickle.loads(data["embeddings"])
        # 根据embedding conf解析embedding
        embeddings_allowed = {}
        if self._embedding_conf and isinstance(self._embedding_conf, dict):
            for key in self._embedding_conf.keys():
                if key in embeddings.keys():
                    embeddings_allowed.update({key: embeddings[key]})
                else:
                    embeddings_allowed.update({key: np.zeros(self._embedding_conf[key])})
        else:
            embeddings_allowed = embeddings
        embeddings_input = {}
        for k, v in embeddings_allowed.items():
            if isinstance(v, list) and len(v) == self._embedding_conf[k]:
                embeddings_input.update({k: torch.tensor(v, dtype=torch.float32)})
            else:
                embeddings_input.update({k: torch.zeros(self._embedding_conf[k])})
        # 数值特征
        auto_dis_input, feature_dense = self.process_feature_dense(features=strategy)
        asr_input = self.process_text(asr)
        # 人审召回相关的label
        label = labels.get("label", 0)  # 处罚维度label
        # 更新传入模型的输入
        feature_data.update(
            {
                "auto_dis_input": auto_dis_input,
                "feature_dense": feature_dense,
                "input_ids": asr_input["input_ids"],
                "attention_mask": asr_input["attention_mask"],
                "token_type_ids": asr_input["token_type_ids"],
                "label": self._process_label(label),
            }
        )
        feature_data.update(embeddings_input)
        if self._save_extra:
            extra = {"object_id": data["object_id"]}
            extra.update(features["context"])
            extra.update(
                {
                    "label": int(label),
                    "uv": int(data["uv"]),
                    "online_score": data["online_score"],
                    "verify_reason": data["verify_reason"],
                }
            )
            extra_str = json.dumps(extra, ensure_ascii=False)
            try:
                feature_data.update({"extra": extra_str})
            except Exception as e:
                get_logger().error("{}, fail to add extra: {}".format(e, extra))
        return feature_data


@FEATURE_PROVIDERS.register_module()
class EcomLiveGandalfParquetAutoDisMtlFeatureProvider(GandalfParquetFeatureProvider):
    def __init__(
        self,
        feature_num,
        max_len=512,
        use_high_precision=False,
        filtered_tags=None,
        slot_mask=None,
        feature_norm_info=None,
        embedding_conf=None,
        tokenizer_path=None,
        save_extra=False,
        eval_mode=False,
        trace_mode=False,
        **kwargs,
    ):
        super(EcomLiveGandalfParquetAutoDisMtlFeatureProvider, self).__init__()
        self._slot_mask = slot_mask
        self._feature_num = feature_num
        self._use_high_precision = use_high_precision
        self._filtered_tags = filtered_tags
        self._feature_norm_info = feature_norm_info
        self._embedding_conf = embedding_conf
        self._feature_input_num = feature_num - len(slot_mask)
        self._active_slot = [i for i in range(self._feature_num) if i not in self._slot_mask]
        # self._text_tokenizer = AutoTokenizer.from_pretrained(self.asr_model_name, return_tensors="pt", max_length=512)     # noqa: E501
        self._text_tokenizer = DebertaTokenizer(f"{tokenizer_path}/vocab.txt", max_len=max_len)
        self._save_extra = save_extra
        self._trace_mode = trace_mode

    def process_feature_dense(self, features):
        # 加载预处理参数
        active_slot = torch.tensor(self._active_slot, dtype=torch.long).reshape(len(self._active_slot))
        compressed_min = [self._feature_norm_info.get(str(slot_id), [0, 1])[0] for slot_id in self._active_slot]
        compressed_max = [self._feature_norm_info.get(str(slot_id), [0, 1])[1] for slot_id in self._active_slot]
        compressed_min = torch.tensor(compressed_min, dtype=torch.float32).reshape(-1)
        compressed_max = torch.tensor(compressed_max, dtype=torch.float32).reshape(-1)
        compressed_range = compressed_max - compressed_min
        # 并行特征预处理
        features = torch.tensor(features, dtype=torch.float32)
        feature_dense = features[active_slot]
        feature_dense_norm = (feature_dense - compressed_min) / compressed_range
        feature_dense_norm[feature_dense == -1] = 0.0  # 特征为-1, norm值为0
        feature_dense_norm[feature_dense.isnan()] = 0.0  # 特征缺失, norm值为0
        feature_dense_norm = torch.clamp(feature_dense_norm, min=0.0, max=1.0)
        auto_dis_input_list = [
            feature_dense_norm,
            feature_dense_norm * feature_dense_norm,
            torch.sqrt(feature_dense_norm),
        ]
        auto_dis_input = torch.stack(auto_dis_input_list, dim=1)
        return auto_dis_input, feature_dense_norm

    def process_text(self, text):
        # 加载预处理参数
        asr_inputs = self._text_tokenizer(text)
        return asr_inputs

    def process(self, data):
        feature_data = {}
        # 数据mask:是否mask高准数据流
        source = data.get("source", "origin")
        if source == "high_precision" and not self._use_high_precision:
            return None
        # 标签mask:是否mask特定标签数据
        if self._filtered_tags and data.get("verify_reason", ""):
            if len(re.findall(self._filtered_tags, data.get("verify_reason", ""))) > 0:
                # get_logger().info("Mask data by filtered tags:{}".format(data.get('verify_reason','')))
                return None
        # parquet数据主要包含labels,features,contents,embeddings字段，而且均已经序列化了
        labels = pickle.loads(data["labels"])
        # numerical features
        features = pickle.loads(data["features"])
        strategy = list(features["strategy"].values())
        # content features
        contents = pickle.loads(data["contents"])
        asr = contents["asr"]
        # embedding features
        embeddings = pickle.loads(data["embeddings"])
        # 根据embedding conf解析embedding
        embeddings_allowed = {}
        if self._embedding_conf and isinstance(self._embedding_conf, dict):
            for key in self._embedding_conf.keys():
                if key in embeddings.keys():
                    embeddings_allowed.update({key: embeddings[key]})
                else:
                    embeddings_allowed.update({key: np.zeros(self._embedding_conf[key])})
        else:
            embeddings_allowed = embeddings
        embeddings_input = {}
        for k, v in embeddings_allowed.items():
            if isinstance(v, list) and len(v) == self._embedding_conf[k]:
                embeddings_input.update({k: torch.tensor(v, dtype=torch.float32)})
            else:
                embeddings_input.update({k: torch.zeros(self._embedding_conf[k])})
        # 数值特征
        auto_dis_input, feature_dense = self.process_feature_dense(features=strategy)
        asr_input = self.process_text(asr)
        # 进审相关label
        censor_label = 1 if source == "origin" else 0  # 进审维度label
        # 人审召回相关的label
        reject_label = labels.get("label", 0)  # 处罚维度label
        # 更新传入模型的输入
        feature_data.update(
            {
                "auto_dis_input": auto_dis_input,
                "feature_dense": feature_dense,
                "input_ids": asr_input["input_ids"],
                "attention_mask": asr_input["attention_mask"],
                "token_type_ids": asr_input["token_type_ids"],
                "censor_label": self._process_label(censor_label),
                "reject_label": self._process_label(reject_label),
            }
        )
        feature_data.update(embeddings_input)
        if self._save_extra:
            extra = {"object_id": data["object_id"]}
            extra.update(features["context"])
            extra.update(
                {
                    "censor_label": int(censor_label),
                    "reject_label": int(reject_label),
                    "uv": int(data["uv"]),
                    "online_score": data["online_score"],
                    "verify_reason": data["verify_reason"],
                }
            )
            extra_str = json.dumps(extra, ensure_ascii=False)
            try:
                feature_data.update({"extra": extra_str})
            except Exception as e:
                get_logger().error("{}, fail to add extra: {}".format(e, extra))
        return feature_data


@DATASETS.register_module()
class EcomLiveGandalfParquetAutoDisCruiseDataModule(GandalfParquetCruiseDataModule):
    def __init__(self, dataset, feature_provider, data_factory, type=None):
        super(EcomLiveGandalfParquetAutoDisCruiseDataModule, self).__init__()
        self.save_hparams()

    def setup(self, stage: Optional[str] = None) -> None:
        # print(self.hparams)
        # print('self.hparams.dataset',type(self.hparams.dataset),self.hparams.dataset)
        self.dataset = Dict(self.hparams.dataset)
        self.feature_provider = Dict(self.hparams.feature_provider)
        self.data_factory = Dict(self.hparams.data_factory)
        self.total_cfg = Dict(
            {
                "dataset": self.dataset,
                "feature_provider": self.feature_provider,
                "data_factory": self.data_factory,
            }
        )
        # self.train_predefined_steps = 'max' if self.data_factory.get('train_max_iteration',-1) == -1 else 'max'
        # self.val_predefined_steps = 'max' if self.data_factory.get('val_max_iteration',-1) == -1 else 'max'

    def train_dataloader(self):
        return self.create_cruise_dataloader(
            self.total_cfg,
            data_input_dir=self.dataset.input_dir,
            data_folder=self.dataset.train_folder,
            arg_dict=self.data_factory,
            mode="train",
        )

    def val_dataloader(self):
        return self.create_cruise_dataloader(
            self.total_cfg,
            data_input_dir=self.dataset.val_input_dir,
            data_folder=self.dataset.val_folder,
            arg_dict=self.data_factory,
            mode="val",
        )

    def test_dataloader(self):
        return self.create_cruise_dataloader(
            self.total_cfg,
            data_input_dir=self.dataset.test_input_dir,
            data_folder=self.dataset.test_folder,
            arg_dict=self.data_factory,
            mode="test",
        )
