"""Unsupervised datamodule for GPT pretraining"""
import collections
import logging
import os
import re
import tempfile

# from transformers import LlamaTokenizer
import warnings
from math import ceil
from typing import List, Union

import torch
from cruise import CruiseDataModule, last_cli
from cruise.data_module import DistributedCruiseDataLoader
from cruise.data_module.gpu_wrapper import GPUPrefetcher
from cruise.data_module.tools import create_dataloader_by_cfg
from cruise.utilities import DIST_ENV
from cruise.utilities.hdfs_io import hcopy, hlist_files
from pydantic import NoneIsAllowedError
from torch.utils.data._utils.collate import default_collate
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from ..tokenization import CasterTokenizer

# try:
#     from promptsource.templates import DatasetTemplates, Template
# except ImportError:
#     warnings.warn('failed to load prompt source', ImportWarning)


class NextWordDataProcessor:
    def __init__(
        self,
        tokenizer: str,
        max_seq_len: int,
        finetune_type_is_qa: bool,
        drop_last: bool = False,
        use_loss_mask: bool = False,
        stride=-1,
        **kwargs,
    ):
        if not isinstance(tokenizer, str):
            # from created tokenizer object
            self.tokenizer = tokenizer
        else:
            if "alpaca-native" in tokenizer:
                from transformers import LlamaTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, max_len=-1)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, max_len=-1)
        self.max_seq_len = max_seq_len
        self.finetune_type_is_qa = finetune_type_is_qa
        self.drop_last = drop_last
        self.use_loss_mask = use_loss_mask
        self.stride = stride

        # We will automatically convert token list to tensor
        kwargs.pop("return_tensors", None)
        self.kwargs = kwargs

    def transform(self, data_dict):
        if self.finetune_type_is_qa:
            data_dict["text"] = data_dict["question"] + data_dict["answer"]

        prompt = data_dict["text"]
        prompt_outputs = self.tokenizer(prompt, **self.kwargs)

        context_enc = [self.tokenizer.bos_token_id] + prompt_outputs["input_ids"] + [self.tokenizer.eos_token_id]
        context_mask = [1] + prompt_outputs["attention_mask"] + [1]

        full_enc = context_enc
        full_mask = context_mask
        if self.drop_last:
            input_ids = full_enc[-(self.max_seq_len) :]
            attention_mask = full_mask[-(self.max_seq_len) :]

            input_len = len(input_ids)
            input_ids = [self.tokenizer.pad_token_id] * (self.max_seq_len - input_len) + input_ids
            attention_mask = [0] * (self.max_seq_len - input_len) + attention_mask
        else:
            input_ids = full_enc
            attention_mask = full_mask

        text_dict = {
            "input_ids": [input_ids],
            "attention_mask": [attention_mask],
        }

        if self.use_loss_mask:
            if self.finetune_type_is_qa:
                prompt_output_question = self.tokenizer(data_dict["question"], **self.kwargs)
                prompt_output_answer = self.tokenizer(data_dict["answer"], **self.kwargs)
                text_dict["input_ids"] = [
                    [self.tokenizer.bos_token_id]
                    + prompt_output_question["input_ids"]
                    + prompt_output_answer["input_ids"]
                    + [self.tokenizer.eos_token_id]
                ]
                text_dict["attention_mask"] = [
                    [1] + prompt_output_question["attention_mask"] + prompt_output_answer["attention_mask"] + [1]
                ]
                if "is_instruct" not in text_dict:
                    text_dict["loss_mask"] = [
                        [0]
                        + [0] * len(prompt_output_question["attention_mask"])
                        + [1] * len(prompt_output_answer["attention_mask"])
                        + [1]
                    ]
                else:
                    if text_dict["is_instruct"] == False:
                        text_dict["loss_mask"] = [
                            [0]
                            + [0] * len(prompt_output_question["attention_mask"])
                            + [1] * len(prompt_output_answer["attention_mask"])
                            + [1]
                        ]
                    else:
                        text_dict["loss_mask"] = [
                            [1]
                            + [1] * len(prompt_output_question["attention_mask"])
                            + [1] * len(prompt_output_answer["attention_mask"])
                            + [1]
                        ]

            else:
                assert KeyError

        # print('text: ')
        # print(data_dict['text'])
        # print(self.group_texts(text_dict))

        return self.group_texts(text_dict)

    def batch_transform(self, batch_data):
        return default_collate(batch_data)

    def group_texts(self, examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        ori_total_length = len(concatenated_examples[list(examples.keys())[0]])

        mod_rst = ori_total_length % self.max_seq_len
        total_length = (ori_total_length // self.max_seq_len) * self.max_seq_len

        # print(f'total length is {total_length}')
        # print(f'mod_rst is: {mod_rst}')
        if mod_rst:
            if total_length < self.max_seq_len:
                for key in concatenated_examples:
                    if "mask" in key:
                        pad_token_id = 0
                    else:
                        pad_token_id = self.tokenizer.pad_token_id
                    concatenated_examples[key] = [pad_token_id] * (
                        self.max_seq_len - ori_total_length
                    ) + concatenated_examples[key]
                total_length = self.max_seq_len
            elif not self.drop_last:
                if self.stride != -1:
                    n_subseqs = ceil((ori_total_length - self.max_seq_len) / self.stride) + 1
                    for key in concatenated_examples:
                        subseqs = [
                            concatenated_examples[key][i * self.stride : i * self.stride + self.max_seq_len]
                            for i in range(n_subseqs)
                        ]
                        pad_token_id = 0 if "mask" in key else self.tokenizer.pad_token_id
                        concatenated_examples[key] = []
                        for i in range(n_subseqs):
                            if len(subseqs[i]) < self.max_seq_len:
                                # subseqs[i].extend([pad_token_id] * (self.max_seq_len-len(subseqs[i])) )
                                subseqs[i][:0] = [pad_token_id] * (self.max_seq_len - len(subseqs[i]))
                            concatenated_examples[key].extend(subseqs[i])
                    total_length = len(concatenated_examples[list(examples.keys())[0]])
                else:
                    for key in concatenated_examples:
                        concatenated_examples[key] = (
                            concatenated_examples[key][:total_length] + concatenated_examples[key][-self.max_seq_len :]
                        )
                    total_length = total_length + self.max_seq_len

        # Split by chunks of max_len.
        outputs = []
        for i in range(0, total_length, self.max_seq_len):
            result = {k: torch.as_tensor(t[i : i + self.max_seq_len]) for k, t in concatenated_examples.items()}
            outputs.append(result)

        # print(f' len is {len(outputs)}')
        # print(outputs)

        return outputs


class ZeroShotGPTDatamodule(CruiseDataModule):
    def __init__(
        self,
        train_path: str = "",
        val_path: str = "",
        train_size: int = -1,  # num of tokens when dyn_bsz True, else num of docs
        train_batch_size: int = 4,
        train_num_workers: int = 4,
        val_batch_size: int = 32,
        val_num_workers: int = 1,
        max_seq_len: int = 2048,
        tokenizer: str = "",
        gpu_prefetch: bool = False,
        dyn_bsz: bool = False,
        dyn_bsz_margin: float = 0.0,
        stride: int = 896,
        warmup_step_rate: float = -1,
        bsz_warmup: bool = False,
        bsz_warmup_rate: float = 0.02,
        finetune_type_is_qa=False,
        use_loss_mask=False,
        from_hf_tokenizer: bool = False,
        hf_tokenizer_use_fast: bool = False,
        drop_last: bool = False,
    ):
        super().__init__()
        self.save_hparams()

        if self.hparams.bsz_warmup:
            global_config = last_cli().hparams
            warmup_rate = global_config["trainer"]["optimizer_kwargs"]["scheduler"]["params"]["warmup_step_rate"]
            max_epochs = global_config["trainer"]["max_epochs"]
            self.hparams.warmup_step_rate = warmup_rate * max_epochs

        self.templates = None
        self.tokenizer = None
        self.processor = None

    def local_rank_zero_prepare(self) -> None:
        if self.hparams.tokenizer.startswith("hdfs"):
            # try download it to local once per node and load it in setup
            tmp_dir = os.path.join(tempfile.gettempdir(), os.path.basename(self.hparams.tokenizer))
            logging.info(f"tokenizer tmp path is {tmp_dir}")
            if not os.path.exists(tmp_dir):
                logging.info("tokenizer tmp path does not exist, hcopy...")
                hcopy(self.hparams.tokenizer, tmp_dir)
        else:
            logging.info(f"Prefetching HF tokenizers {self.hparams.tokenizer} on local rank zero...")
            from transformers import AutoTokenizer

            if "alpaca-native" in self.hparams.tokenizer:
                from transformers import LlamaTokenizer

                AutoTokenizer.from_pretrained(self.hparams.tokenizer)
            else:
                AutoTokenizer.from_pretrained(self.hparams.tokenizer)

    def setup(self):
        if self.hparams.tokenizer.startswith("hdfs"):
            # try download it to local once per node and load it in setup
            tmp_dir = os.path.join(tempfile.gettempdir(), os.path.basename(self.hparams.tokenizer))
            if self.hparams.from_hf_tokenizer:
                if "alpaca-native" in tmp_dir:
                    from transformers import LlamaTokenizer

                    self.tokenizer = AutoTokenizer.from_pretrained(
                        tmp_dir, use_fast=self.hparams.hf_tokenizer_use_fast
                    )
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        tmp_dir, use_fast=self.hparams.hf_tokenizer_use_fast
                    )
            else:
                self.tokenizer = CasterTokenizer.from_pretrained(tmp_dir, max_len=-1)
        else:
            if "alpaca-native" in self.hparams.tokenizer:
                from transformers import LlamaTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer, max_len=-1)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer, max_len=-1)

        self.finetune_type_is_qa = self.hparams.finetune_type_is_qa
        self.use_loss_mask = self.hparams.use_loss_mask

        self.processor = NextWordDataProcessor(
            tokenizer=self.tokenizer if self.tokenizer is not None else self.hparams.tokenizer,
            max_seq_len=self.hparams.max_seq_len,
            finetune_type_is_qa=self.finetune_type_is_qa,
            drop_last=self.hparams.drop_last,
            use_loss_mask=self.hparams.use_loss_mask,
            stride=self.hparams.stride,
        )

    def get_train_steps(self):
        if self.hparams.dyn_bsz:  # 按照token bsz算
            eff_token_rate = 0.95
            train_size = int(
                self.hparams.train_size
                * (1 + self.hparams.bsz_warmup_rate / 2)
                / eff_token_rate
                * (self.hparams.max_seq_len / self.hparams.stride)
            )
            self.rank_zero_info(f"Estimated Train Size: {train_size}")
            train_steps = int(
                train_size // (self.hparams.train_batch_size * DIST_ENV.world_size * self.hparams.max_seq_len)
            )
        else:
            train_steps = self.hparams.train_size // (self.hparams.train_batch_size * DIST_ENV.world_size)
        assert (
            train_steps > 0
        ), f"train_size={self.hparams.train_size} may be too small to split to batch_size * world_size"
        return train_steps

    def train_dataloader(self):
        if not self.hparams.train_path:
            return iter([])
        train_steps = -1
        if self.hparams.train_size > 0:
            train_steps = self.get_train_steps()
        self.train_steps = train_steps

        train_files = [x for x in hlist_files([self.hparams.train_path]) if x.endswith(".parquet")]

        if self.hparams.bsz_warmup:
            assert self.hparams.warmup_step_rate > 0
        train_files = sorted(train_files)

        self.rank_zero_info(f"Fetched {len(train_files)} train files.")

        loader = DistributedCruiseDataLoader(
            data_sources=[train_files],
            batch_sizes=[self.hparams.train_batch_size],
            num_workers=self.hparams.train_num_workers,
            predefined_steps=train_steps,
            source_types=["parquet"],
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            parquet_cache_on=True,
            keys_or_columns=None,
            num_readers=[1],
            decode_fn_list=None,
            processor=self.processor,
            transform_output_many=True,
            dyn_bsz=self.hparams.dyn_bsz,
            dyn_bsz_margin=self.hparams.dyn_bsz_margin,
            num_warmup_steps=int(self.hparams.bsz_warmup_rate * train_steps) if self.hparams.bsz_warmup else -1,
        )
        if self.hparams.gpu_prefetch:
            loader = GPUPrefetcher(loader)
        return loader

    def val_dataloader(self):
        if not self.hparams.val_path:
            return iter([])
        val_steps = -1
        val_files = [x for x in hlist_files([self.hparams.val_path]) if x.endswith(".parquet")]
        self.rank_zero_info(f"Fetched {len(val_files)} val files.")
        loader = DistributedCruiseDataLoader(
            data_sources=[val_files],
            batch_sizes=[self.hparams.val_batch_size],
            num_workers=self.hparams.val_num_workers,
            predefined_steps=val_steps,
            source_types=["parquet"],
            shuffle=False,
            drop_last=self.hparams.drop_last,
            pin_memory=True,
            parquet_cache_on=True,
            keys_or_columns=None,
            num_readers=[1],
            decode_fn_list=None,
            processor=self.processor,
            transform_output_many=True,
        )
        if self.hparams.gpu_prefetch:
            loader = GPUPrefetcher(loader)
        return loader
