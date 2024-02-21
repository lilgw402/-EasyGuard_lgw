# -*- coding: utf-8 -*-

from typing import List, Union

import torch
from torch.utils.data._utils.collate import default_collate
from transformers import AutoTokenizer

try:
    import cruise
except ImportError:
    print(
        "[ERROR] cruise is not installed! Please refer this doc: https://bytedance.feishu.cn/wiki/wikcnGP7yzZAuKpPfL6jRJKl2ag"
    )

from cruise.data_module import CruiseDataModule
from cruise.data_module.cruise_loader import DistributedCruiseDataLoader
from cruise.utilities.hdfs_io import hlist_files


class TextPreProcessor:
    def __init__(
        self,
        x_key,
        y_key,
        region_key,
        pre_tokenize,  # mlm_probability, cl_enable, cla_task_enable, category_key,
        max_len,
        tokenizer,
    ):
        self._x_key = x_key
        self._y_key = y_key
        self._region_key = region_key
        self._pre_tokenize = pre_tokenize
        # self._mlm_probability = mlm_probability
        # self._cl_enable = cl_enable
        # self._cla_task_enable = cla_task_enable
        # self._category_key = category_key
        self._max_len = max_len
        self._tokenizer = tokenizer

    def transform(self, data_dict: dict):
        # == 文本 ==
        if not self._pre_tokenize:  # do not pre tokenize
            text = data_dict.get(self._x_key, " ")
            text_token = self._tokenizer(
                text,
                padding="max_length",
                max_length=self._max_len,
                truncation=True,
            )
            if "token_type_ids" not in self._tokenizer.model_input_names:
                text_token["token_type_ids"] = [0] * self._max_len
        else:
            text_token = data_dict[self._x_key]
            text_token[0] = self._tokenizer.cls_token
            text_token[-1] = self._tokenizer.sep_token
            text_token_ids = self._tokenizer.convert_tokens_to_ids(text_token)
            text_token["input_ids"] = text_token_ids
            text_token["attention_mask"] = [1] * len(text_token_ids[: self._max_len]) + [0] * (
                self._max_len - len(text_token_ids)
            )
            text_token["token_type_ids"] = [0] * self._max_len

        language = data_dict[self._region_key]

        input_dict = {
            "language": language,
            "input_ids": torch.Tensor(text_token["input_ids"]).long(),
            "attention_mask": torch.Tensor(text_token["attention_mask"]).long(),
            "token_type_ids": torch.Tensor(text_token["token_type_ids"]).long(),
        }

        # == 标签 ==
        label = int(data_dict[self._y_key])
        input_dict["labels"] = torch.tensor(label)

        return input_dict

    def batch_transform(self, batch_data):
        # batch_data: List[Dict[modal, modal_value]]
        out_batch = {}
        keys = ("input_ids", "attention_mask", "token_type_ids", "labels")

        for k in keys:
            out_batch[k] = default_collate([data[k] for data in batch_data])

        return out_batch


class SequenceClassificationData(CruiseDataModule):
    def __init__(
        self,
        train_batch_size: int = 16,
        val_batch_size: int = 16,
        train_file: str = "hdfs://harunava/home/byte_magellan_va/user/jiangjunjun.happy/xtreme_ecom/ansa_train.json",
        val_file: str = "hdfs://harunava/home/byte_magellan_va/user/jiangjunjun.happy/xtreme_ecom/leaderboard/ansa_test.json",
        data_size: int = 200000000,
        val_step: int = 500,
        num_workers: int = 1,
        tokenizer: str = "microsoft/mdeberta-v3-base",
        x_key: str = "text",
        y_key: str = "label",
        region_key: str = "country",
        pre_tokenize: bool = False,
        #  mlm_probability: float = 0.15,
        #  cl_enable: bool = False,
        #  cla_task_enable: bool = False,
        max_len: int = 256,
    ):
        super().__init__()
        self.save_hparams()

    def local_rank_zero_prepare(self) -> None:
        # download the tokenizer once per node
        AutoTokenizer.from_pretrained(self.hparams.tokenizer)

    def setup(self, stage) -> None:
        # paths = self.hparams.paths
        # if isinstance(paths, str):
        #     paths = [paths]
        # # split train/val
        # files = hlist_files(paths)
        # if not files:
        #     raise RuntimeError(f"No valid files can be found matching `paths`: {paths}")
        # files = sorted(files)
        # # use the last file as validation
        # self.train_files = files
        # self.val_files = files[0:16]
        self.train_files = [self.hparams.train_file]
        self.val_files = [self.hparams.train_file]

        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer)

    def train_dataloader(self):
        return DistributedCruiseDataLoader(
            data_sources=[self.train_files],
            keys_or_columns=[None],
            batch_sizes=[self.hparams.train_batch_size],
            num_workers=self.hparams.num_workers,
            num_readers=[1],
            decode_fn_list=[[]],
            processor=TextPreProcessor(
                self.hparams.x_key,
                self.hparams.y_key,
                self.hparams.region_key,
                self.hparams.pre_tokenize,
                # self.hparams.mlm_probability, self.hparams.cl_enable, self.hparams.cla_task_enable, self.hparams.category_key,
                self.hparams.max_len,
                self.tokenizer,
            ),
            predefined_steps=self.hparams.data_size // self.hparams.train_batch_size // self.trainer.world_size,
            source_types=["jsonl"],
            shuffle=True,
        )

    def val_dataloader(self):
        return DistributedCruiseDataLoader(
            data_sources=[self.val_files],
            keys_or_columns=[None],
            batch_sizes=[self.hparams.val_batch_size],
            num_workers=self.hparams.num_workers,
            num_readers=[1],
            decode_fn_list=[[]],
            processor=TextPreProcessor(
                self.hparams.x_key,
                self.hparams.y_key,
                self.hparams.region_key,
                self.hparams.pre_tokenize,
                # self.hparams.mlm_probability, self.hparams.cl_enable, self.hparams.cla_task_enable, self.hparams.category_key,
                self.hparams.max_len,
                self.tokenizer,
            ),
            predefined_steps=self.hparams.val_step,
            source_types=["jsonl"],
            shuffle=False,
        )
