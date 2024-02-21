"""Unsupervised datamodule for GPT pretraining"""
import logging
import os
import tempfile
from math import ceil
from typing import List, Union

import torch
from cruise import CruiseDataModule, last_cli
from cruise.data_module.gpu_wrapper import GPUPrefetcher
from cruise.utilities import DIST_ENV
from cruise.utilities.hdfs_io import hcopy, hlist_files
from torch.utils.data._utils.collate import default_collate
from transformers import AutoTokenizer

from ..tokenization import CasterTokenizer

# from cruise.data_module import DistributedCruiseDataLoader
from .cruise_loader import DistributedCruiseDataLoader


class RawTextProcessor:
    r"""
    Args:
        tokenizer: the name of the pretrained tokenizer, e.g., "bigscience/bloom"
        text_keys: keys that contains text as values in the input.
        max_seq_len: max length that the model accept, if data is not enough,
                     pad_token_id will be used.
        drop_last: if text length is not divisible by max_seq_len, set this
                   field to False will pad the remainder.
        stride: if 'slidng_window' is not -1, the text will be sampled with sliding window of stride 'stride'.
    """

    def __init__(
        self,
        tokenizer: str,
        text_keys: Union[str, List[str]],
        max_seq_len: int,
        drop_last: bool = False,
        stride=-1,
        **kwargs,
    ):
        if not isinstance(text_keys, list):
            text_keys = [text_keys]
        self.text_keys = text_keys
        if not isinstance(tokenizer, str):
            # from created tokenizer object
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_seq_len = max_seq_len
        self.drop_last = drop_last
        self.stride = stride
        # We will automatically convert token list to tensor
        kwargs.pop("return_tensors", None)
        self.kwargs = kwargs

    def transform(self, data_dict):
        text_dict = {}
        for key in self.text_keys:
            text_output = self.tokenizer(data_dict[key], **self.kwargs)
            for k, v in text_output.items():
                if k not in text_dict:
                    text_dict[k] = [v]
                else:
                    text_dict[k].append(v)
        # append EOS token到末尾, 中途不加token，这个处理方法待定
        for k, v in text_dict.items():
            if "mask" in k:
                eos_token_id = 1  # EOS mask is still 1
            else:
                eos_token_id = self.tokenizer.eos_token_id
            v[-1] += [eos_token_id]
        return self.group_texts(text_dict)

    def batch_transform(self, batch_data):
        return default_collate(batch_data)

    def group_texts(self, examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        ori_total_length = len(concatenated_examples[list(examples.keys())[0]])

        mod_rst = ori_total_length % self.max_seq_len
        total_length = (ori_total_length // self.max_seq_len) * self.max_seq_len
        if mod_rst:
            if total_length < self.max_seq_len:
                for key in concatenated_examples:
                    if "mask" in key:
                        pad_token_id = 0
                    else:
                        pad_token_id = self.tokenizer.pad_token_id
                    concatenated_examples[key] = concatenated_examples[key] + [pad_token_id] * (
                        self.max_seq_len - total_length
                    )
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
                                subseqs[i].extend([pad_token_id] * (self.max_seq_len - len(subseqs[i])))
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
            # result["labels"] = result["input_ids"].clone()
            outputs.append(result)
        return outputs


class UnsupGPTDatamodule(CruiseDataModule):
    """GPT pretrain dataset module."""

    def __init__(
        self,
        train_path: Union[List[str], str] = "",
        val_path: str = "",
        train_size: int = 300_000_000_000,  # num of tokens when dyn_bsz True, else num of docs
        train_batch_size: int = 32,
        train_num_workers: int = 1,
        val_batch_size: int = 32,
        val_num_workers: int = 1,
        max_seq_len: int = 1024,
        text_keys: List[str] = ["content_split"],
        tokenizer: str = "",
        gpu_prefetch: bool = False,
        dyn_bsz: bool = False,
        dyn_bsz_margin: float = 0.0,
        stride: int = 896,
        warmup_step_rate: float = -1,
        tokenizer_type: str = "caster",
        bsz_warmup: bool = False,
        bsz_warmup_rate: float = 0.02,
    ):
        super().__init__()
        self.save_hparams()

        if self.hparams.bsz_warmup:
            global_config = last_cli().hparams
            warmup_rate = global_config["trainer"]["optimizer_kwargs"]["scheduler"]["params"]["warmup_step_rate"]
            max_epochs = global_config["trainer"]["max_epochs"]
            self.hparams.warmup_step_rate = warmup_rate * max_epochs
        self.tokenizer = None

    def local_rank_zero_prepare(self) -> None:
        if self.hparams.tokenizer.startswith("hdfs"):
            # try download it to local once per node and load it in setup
            tmp_dir = os.path.join(tempfile.gettempdir(), os.path.basename(self.hparams.tokenizer))
            hcopy(self.hparams.tokenizer, tmp_dir)
        else:
            logging.info(f"Prefetching HF tokenizers {self.hparams.tokenizer} on local rank zero...")
            AutoTokenizer.from_pretrained(self.hparams.tokenizer)

    def setup(self):
        if self.hparams.tokenizer.startswith("hdfs"):
            # try download it to local once per node and load it in setup
            tmp_dir = os.path.join(tempfile.gettempdir(), os.path.basename(self.hparams.tokenizer))
            if self.hparams.tokenizer_type == "caster":
                self.tokenizer = CasterTokenizer.from_pretrained(tmp_dir, max_len=-1)
            elif self.hparams.tokenizer_type == "bbpe":
                self.tokenizer = AutoTokenizer.from_pretrained(tmp_dir)
            else:
                raise NotImplementedError
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer, max_len=-1)

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
        train_steps = -1
        if self.hparams.train_size > 0:
            train_steps = self.get_train_steps()
        self.train_steps = train_steps
        source_type = "parquet"
        if isinstance(self.hparams.train_path, (list, tuple)):
            # TODO: utilize cruise parser to handle this, currently just make it work with simple trick
            # To parse a list of folders with json/snappy files, does not support wildcard matching
            train_files = []
            for source in self.hparams.train_path:
                path = os.path.dirname(source)
                ext = os.path.splitext(source)[-1]
                files = [x for x in hlist_files([path]) if x.endswith(ext)]
                train_files += files
            source_type = "jsonl"
        else:
            train_files = [x for x in hlist_files([self.hparams.train_path]) if x.endswith(".parquet")]
        if self.hparams.bsz_warmup:
            assert self.hparams.warmup_step_rate > 0
        train_files = sorted(train_files)
        self.rank_zero_info(f"Fetched {len(train_files)} training files.")
        if self.hparams.tokenizer_type == "bbpe":
            tokenizer_kwargs = {"return_token_type_ids": False}
        else:
            tokenizer_kwargs = {}
        loader = DistributedCruiseDataLoader(
            data_sources=[train_files],
            batch_sizes=[self.hparams.train_batch_size],
            num_workers=self.hparams.train_num_workers,
            predefined_steps=train_steps,
            source_types=[source_type],
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            parquet_cache_on=True,
            keys_or_columns=None,
            num_readers=[1],
            decode_fn_list=None,
            processor=RawTextProcessor(
                tokenizer=self.tokenizer if self.tokenizer is not None else self.hparams.tokenizer,
                text_keys=self.hparams.text_keys,
                max_seq_len=self.hparams.max_seq_len,
                drop_last=False,
                stride=self.hparams.stride,
                **tokenizer_kwargs,
            ),
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
            drop_last=False,
            pin_memory=True,
            parquet_cache_on=True,
            keys_or_columns=None,
            num_readers=[1],
            decode_fn_list=None,
            processor=RawTextProcessor(
                tokenizer=self.tokenizer if self.tokenizer is not None else self.hparams.tokenizer,
                text_keys=self.hparams.text_keys,
                max_seq_len=self.hparams.max_seq_len,
                drop_last=False,
            ),
            transform_output_many=True,
        )
        if self.hparams.gpu_prefetch:
            loader = GPUPrefetcher(loader)
        return loader
