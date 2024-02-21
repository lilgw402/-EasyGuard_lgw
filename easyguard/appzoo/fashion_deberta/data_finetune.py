"""An customizable fashion_deberta example"""
import json
import os
import sys
from typing import List, Union

import torch

try:
    import easyguard
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from cruise.data_module import CruiseDataModule
from cruise.data_module.cruise_loader import DistributedCruiseDataLoader
from cruise.utilities.hdfs_io import hlist_files

from easyguard.core import AutoTokenizer
from easyguard.utils.data_helpers import build_vocab


class TextProcessor:
    def __init__(self, text_label_field, text_field, tokenizer, context_length, vocab):
        self._text_label_field = text_label_field
        self._text_field = text_field
        self._tokenizer = tokenizer
        self._context_length = context_length
        self._vocab = vocab
        self.PAD_IDX = self._vocab["[PAD]"]
        self.SEP_IDX = self._vocab["[SEP]"]
        self.CLS_IDX = self._vocab["[CLS]"]
        self.MASK_IDX = self._vocab["[MASK]"]
        self.cnt = 0

    def transform(self, data_dict: dict):
        # get image by key, order matters
        if not self._text_field in data_dict:
            raise KeyError(f"Unable to find text by keys: {self._text_field} available keys: {data_dict.keys()}")
        text = data_dict.get(self._text_field, "")
        text_token = self._tokenizer.tokenize(text)
        text_token = text_token[: self._context_length - 2]
        text_token_ids = [self._vocab[token] for token in text_token]
        text_token_ids = [self.CLS_IDX] + text_token_ids + [self.SEP_IDX]
        if not self._text_label_field in data_dict:
            raise KeyError(
                f"Unable to find text by keys: {self._text_label_field}, available keys: {data_dict.keys()}"
            )
        label = int(data_dict.get(self._text_label_field))
        return_dict = {
            "input_ids": text_token_ids,
            "classification_labels": int(label),
        }
        return return_dict

    def batch_transform(self, batch_data):
        # batch_data: List[Dict[modal, modal_value]]
        input_ids = []
        input_mask = []
        input_segment_ids = []
        classification_labels = []
        max_len = self._context_length

        for ib, ibatch in enumerate(batch_data):
            input_ids.append(ibatch["input_ids"][:max_len] + [self.PAD_IDX] * (max_len - len(ibatch["input_ids"])))
            input_mask.append([1] * len(ibatch["input_ids"][:max_len]) + [0] * (max_len - len(ibatch["input_ids"])))
            input_segment_ids.append([0] * max_len)

            classification_labels.append(ibatch["classification_labels"])

        input_ids = torch.tensor(input_ids)
        input_mask = torch.tensor(input_mask)
        input_segment_ids = torch.tensor(input_segment_ids)
        classification_labels = torch.tensor(classification_labels)

        res = {
            "input_ids": input_ids,
            "input_masks": input_mask,
            "input_segment_ids": input_segment_ids,
            "classification_labels": classification_labels,
        }
        return res


class FashionDataFtModule(CruiseDataModule):
    def __init__(
        self,
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        train_paths: Union[
            str, List[str]
        ] = "hdfs://haruna/home/byte_ecom_govern/user/yangzheming/chinese/common_model/benchmark/asr_risk_predict/train/*.jsonl",
        val_paths: Union[
            str, List[str]
        ] = "hdfs://haruna/home/byte_ecom_govern/user/yangzheming/chinese/common_model/benchmark/asr_risk_predict/valid/*.jsonl",
        data_size: int = 10000,
        val_step: int = 100,
        text_label_field: str = "label",
        text_field: str = "text",
        num_workers: int = 1,
        context_length: int = 512,
        vocab_file_path: str = "hdfs://haruna/home/byte_ecom_govern/user/yangzheming/asr_model/zh_deberta_base_l6_emd_20210720/vocab.txt",
        pretrain_model_name: str = "fashion-deberta-asr",
    ):
        super().__init__()
        self.save_hparams()

    def setup(self, stage) -> None:
        train_paths = self.hparams.train_paths
        if isinstance(train_paths, str):
            train_paths = [train_paths]
        val_paths = self.hparams.val_paths
        if val_paths:
            if isinstance(val_paths, str):
                val_paths = [val_paths]
            train_files = hlist_files(train_paths)
            val_files = hlist_files(val_paths)
            if not train_files:
                raise RuntimeError(f"No valid files can be found matching `paths`: {train_paths}")
            if not val_files:
                raise RuntimeError(f"No valid files can be found matching `paths`: {val_paths}")
            self.train_files = train_files
            self.val_files = val_files
        else:
            # split train/val
            files = hlist_files(train_paths)
            if not files:
                raise RuntimeError(f"No valid files can be found matching `paths`: {train_paths}")
            # use the last file as validation
            self.train_files = files[:-2]
            self.val_files = files[-2:]
        self.text_label_field = self.hparams.text_label_field
        self.text_field = self.hparams.text_field

        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrain_model_name)
        self.vocab = build_vocab(self.hparams.vocab_file_path)

    def train_dataloader(self):
        return DistributedCruiseDataLoader(
            data_sources=[self.train_files],
            keys_or_columns=[None],
            batch_sizes=[self.hparams.train_batch_size],
            num_workers=self.hparams.num_workers,
            num_readers=[1],
            decode_fn_list=[[]],
            processor=TextProcessor(
                self.text_label_field,
                self.text_field,
                self.tokenizer,
                self.hparams.context_length,
                self.vocab,
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
            processor=TextProcessor(
                self.text_label_field,
                self.text_field,
                self.tokenizer,
                self.hparams.context_length,
                self.vocab,
            ),
            predefined_steps=self.hparams.val_step,
            source_types=["jsonl"],
            shuffle=False,
        )
