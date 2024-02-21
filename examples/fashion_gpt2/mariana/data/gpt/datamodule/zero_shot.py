"""Unsupervised datamodule for GPT pretraining"""
import collections
import logging
import os
import re
import tempfile
import warnings

import torch
from cruise import CruiseDataModule
from cruise.data_module import DistributedCruiseDataLoader
from cruise.data_module.gpu_wrapper import GPUPrefetcher
from cruise.utilities.hdfs_io import hcopy, hlist_files
from torch.utils.data._utils.collate import default_collate
from transformers import AutoTokenizer, PreTrainedTokenizer

from ..tokenization import CasterTokenizer

try:
    from promptsource.templates import DatasetTemplates, Template
except ImportError:
    warnings.warn("failed to load prompt source", ImportWarning)

from torch._six import string_classes

np_str_obj_array_pattern = re.compile(r"[SaUO]")

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, " "dicts or lists; found {}"
)


def customize_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == "numpy" and elem_type.__name__ != "str_" and elem_type.__name__ != "string_":
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: customize_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        return batch


class MultiChoiceDataProcessor:
    def __init__(
        self,
        tokenizer: str,
        max_seq_len: int,
        prompt_template: Template,
        dataset_name: str,
        **kwargs,
    ):
        if not isinstance(tokenizer, str):
            # from created tokenizer object
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, max_len=-1)
        self.max_seq_len = max_seq_len
        self.prompt_template = prompt_template
        self.dataset_name = dataset_name
        # We will automatically convert token list to tensor
        kwargs.pop("return_tensors", None)
        self.kwargs = kwargs

    def transform(self, data_dict):
        outputs = self.prompt_template.apply(data_dict)
        answer_choices_list = self.prompt_template.get_answer_choices_list(data_dict)
        languages = self.prompt_template.metadata.languages

        prompt, target = None, None
        if len(outputs) >= 2:
            prompt = outputs[0]
            targets = outputs[1]
            target = targets[0].strip()

        target_idx = answer_choices_list.index(target)
        transformed_answer_choice_list = [
            answer_choice if "en" not in languages else " " + answer_choice for answer_choice in answer_choices_list
        ]

        prompt_outputs = self.tokenizer(prompt, **self.kwargs)
        answer_choices_list_outputs = [
            self.tokenizer(answer_choice, **self.kwargs) for answer_choice in transformed_answer_choice_list
        ]

        input_ids, attention_masks, answer_choice_tokens_list, input_lens = (
            [],
            [],
            [],
            [],
        )
        for answer_choice_output in answer_choices_list_outputs:
            context_enc = prompt_outputs["input_ids"]
            context_mask = prompt_outputs["attention_mask"]

            # remove start token: </s> if PreTrainedTokenizer
            if isinstance(self.tokenizer, PreTrainedTokenizer):
                continuation_enc = answer_choice_output["input_ids"][1:]
                continuation_mask = answer_choice_output["attention_mask"][1:]
            else:
                continuation_enc = answer_choice_output["input_ids"]
                continuation_mask = answer_choice_output["attention_mask"]

            full_enc = context_enc + continuation_enc
            full_mask = context_mask + continuation_mask

            input = full_enc[-(self.max_seq_len + 1) :][:-1]
            attention_mask = full_mask[-(self.max_seq_len + 1) :][:-1]
            input_len = len(input)

            input = input + [self.tokenizer.pad_token_id] * (self.max_seq_len - input_len)
            attention_mask = attention_mask + [0] * (self.max_seq_len - input_len)

            input_ids.append(input)
            input_lens.append(input_len)
            attention_masks.append(attention_mask)
            answer_choice_tokens_list.append(continuation_enc)

        outputs = [
            {
                "input_ids": torch.as_tensor(input_ids),
                "attention_mask": torch.as_tensor(attention_masks),
                "target_idx": target_idx,
                "answer_choice_tokens_list": answer_choice_tokens_list,
                "input_lens": input_lens,
                "dataset_name": self.dataset_name,
                "task_name": "multi_choice_task",
            }
        ]

        return outputs

    def batch_transform(self, batch_data):
        return customize_collate(batch_data)


class CtxPplDataProcessor:
    def __init__(
        self,
        tokenizer: str,
        max_seq_len: int,
        prompt_template: Template,
        dataset_name: str,
        **kwargs,
    ):
        if not isinstance(tokenizer, str):
            # from created tokenizer object
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, max_len=-1)
        self.max_seq_len = max_seq_len
        self.prompt_template = prompt_template
        self.dataset_name = dataset_name
        # We will automatically convert token list to tensor
        kwargs.pop("return_tensors", None)
        self.kwargs = kwargs

    def transform(self, data_dict):
        outputs = self.prompt_template.apply(data_dict)
        answer_choices_list = self.prompt_template.get_answer_choices_list(data_dict)
        languages = self.prompt_template.metadata.languages

        prompt, target = None, None
        if len(outputs) >= 2:
            prompt = outputs[0]
            targets = outputs[1]
            target = targets[0].strip()

        target_idx = answer_choices_list.index(target)
        transformed_answer_choice_list = [
            answer_choice if "en" not in languages else " " + answer_choice + " "
            for answer_choice in answer_choices_list
        ]

        # print(f'prompt: {prompt}')
        # print(f'target: {target}')
        # print(f'target_idx: {target_idx}')
        # print(f'answer_choices_list: {answer_choices_list}')

        input_ids, attention_masks, input_lens = [], [], []
        for answer_choice in transformed_answer_choice_list:
            ctx = prompt.replace("##blank##", answer_choice)
            ctx_outputs = self.tokenizer(ctx, **self.kwargs)

            input = ctx_outputs["input_ids"][: self.max_seq_len]
            attention_mask = ctx_outputs["attention_mask"][: self.max_seq_len]
            input_len = len(input)

            input = input + [self.tokenizer.pad_token_id] * (self.max_seq_len - input_len)
            attention_mask = attention_mask + [0] * (self.max_seq_len - input_len)

            input_ids.append(input)
            input_lens.append(input_len)
            attention_masks.append(attention_mask)

        outputs = [
            {
                "input_ids": torch.as_tensor(input_ids),
                "attention_mask": torch.as_tensor(attention_masks),
                "target_idx": target_idx,
                "input_lens": input_lens,
                "dataset_name": self.dataset_name,
                "task_name": "ctx_ppl_task",
            }
        ]

        return outputs

    def batch_transform(self, batch_data):
        return customize_collate(batch_data)


class NextWordDataProcessor:
    def __init__(
        self,
        tokenizer: str,
        max_seq_len: int,
        prompt_template: Template,
        dataset_name: str,
        **kwargs,
    ):
        if not isinstance(tokenizer, str):
            # from created tokenizer object
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, max_len=-1)
        self.max_seq_len = max_seq_len
        self.prompt_template = prompt_template
        self.dataset_name = dataset_name
        # We will automatically convert token list to tensor
        kwargs.pop("return_tensors", None)
        self.kwargs = kwargs

    def transform(self, data_dict):
        outputs = self.prompt_template.apply(data_dict)
        languages = self.prompt_template.metadata.languages

        prompt, target = None, None
        if len(outputs) >= 2:
            prompt = outputs[0]
            targets = outputs[1]

            if "en" in languages:
                target = " " + targets[0].strip()
            else:
                target = targets[0].strip()

        prompt_outputs = self.tokenizer(prompt, **self.kwargs)
        target_outputs = self.tokenizer(target, **self.kwargs)

        context_enc = prompt_outputs["input_ids"]
        context_mask = prompt_outputs["attention_mask"]

        # remove start token: </s> if PreTrainedTokenizer
        if isinstance(self.tokenizer, PreTrainedTokenizer):
            continuation_enc = target_outputs["input_ids"][1:]
            continuation_mask = target_outputs["attention_mask"][1:]
        else:
            continuation_enc = target_outputs["input_ids"]
            continuation_mask = target_outputs["attention_mask"]

        # print(f'prompt: {prompt}')
        # print(f'target: {target}')
        # print(f'context_enc: {context_enc}')
        # print(f'continuation_enc: {continuation_enc}')
        # print(f'target_id_to_tokens: {self.tokenizer.convert_ids_to_tokens(continuation_enc)}')
        # print(f'target_id_to_text: {self.tokenizer._decode(continuation_enc)}')

        full_enc = context_enc + continuation_enc
        full_mask = context_mask + continuation_mask

        input_ids = full_enc[-(self.max_seq_len + 1) :][:-1]
        attention_mask = full_mask[-(self.max_seq_len + 1) :][:-1]
        input_len = len(input_ids)

        input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_seq_len - input_len)
        attention_mask = attention_mask + [0] * (self.max_seq_len - input_len)

        outputs = [
            {
                "input_ids": torch.as_tensor(input_ids),
                "attention_mask": torch.as_tensor(attention_mask),
                "target_tokens_list": continuation_enc,
                "input_lens": input_len,
                "dataset_name": self.dataset_name,
                "task_name": "next_word_prediction",
            }
        ]

        return outputs

    def batch_transform(self, batch_data):
        return customize_collate(batch_data)


class LLMPplDataProcessor:
    def __init__(
        self,
        tokenizer: str,
        max_seq_len: int,
        prompt_template: Template,
        dataset_name: str,
        **kwargs,
    ):
        if not isinstance(tokenizer, str):
            # from created tokenizer object
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, max_len=-1)
        self.max_seq_len = max_seq_len
        self.prompt_template = prompt_template
        self.dataset_name = dataset_name
        # We will automatically convert token list to tensor
        kwargs.pop("return_tensors", None)
        self.kwargs = kwargs

    def transform(self, data_dict):
        prompt = data_dict["text"]
        # print(f'prompt: {prompt}')

        prompt_outputs = self.tokenizer(prompt, **self.kwargs)

        context_enc = prompt_outputs["input_ids"]
        context_mask = prompt_outputs["attention_mask"]

        input_ids = context_enc[-(self.max_seq_len + 1) :][:-1]
        attention_mask = context_mask[-(self.max_seq_len + 1) :][:-1]
        input_len = len(input_ids)

        input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_seq_len - input_len)
        attention_mask = attention_mask + [0] * (self.max_seq_len - input_len)

        outputs = [
            {
                "input_ids": torch.as_tensor(input_ids),
                "attention_mask": torch.as_tensor(attention_mask),
                "dataset_name": self.dataset_name,
                "task_name": "llm_ppl",
            }
        ]

        return outputs

    def batch_transform(self, batch_data):
        return customize_collate(batch_data)


class ZeroShotGPTDatamodule(CruiseDataModule):
    def __init__(
        self,
        val_path: str = "",
        val_batch_size: int = 32,
        val_num_workers: int = 1,
        max_seq_len: int = 2048,
        tokenizer: str = "",
        gpu_prefetch: bool = False,
        dataset_name: str = "",
        subset_name: str = "",
        template_name: str = "",
        from_hf_tokenizer: bool = False,
        hf_tokenizer_use_fast: bool = False,
        drop_last: bool = False,
    ):
        super().__init__()
        self.save_hparams()
        self.templates = None
        self.tokenizer = None
        self.processor = None
        self.prompt_template = None

    def local_rank_zero_prepare(self) -> None:
        if self.hparams.tokenizer.startswith("hdfs"):
            # try download it to local once per node and load it in setup
            tmp_dir = os.path.join(tempfile.gettempdir(), os.path.basename(self.hparams.tokenizer))
            hcopy(self.hparams.tokenizer, tmp_dir)
        else:
            logging.info(f"Prefetching HF tokenizers {self.hparams.tokenizer} on local rank zero...")
            from transformers import AutoTokenizer

            AutoTokenizer.from_pretrained(self.hparams.tokenizer)

    def setup(self):
        if self.hparams.tokenizer.startswith("hdfs"):
            # try download it to local once per node and load it in setup
            tmp_dir = os.path.join(tempfile.gettempdir(), os.path.basename(self.hparams.tokenizer))
            if self.hparams.from_hf_tokenizer:
                self.tokenizer = AutoTokenizer.from_pretrained(tmp_dir, use_fast=self.hparams.hf_tokenizer_use_fast)
            else:
                self.tokenizer = CasterTokenizer.from_pretrained(tmp_dir, max_len=-1)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer, max_len=-1)

        if self.hparams.template_name != "":
            self.templates = DatasetTemplates(self.hparams.dataset_name, self.hparams.subset_name)
            template_name = " ".join(self.hparams.template_name.split("+"))
            self.prompt_template = self.templates[template_name]

        if self.hparams.dataset_name in [
            "ai2_arc",
            "super_glue",
            "hellaswag",
            "story_cloze",
            "logi_qa",
            "piqa",
            "openbookqa",
            "logi_qa_zh",
            "piqa_zh",
            "openbookqa_zh",
        ] or self.hparams.subset_name in ["ocnli"]:
            self.processor = MultiChoiceDataProcessor(
                tokenizer=self.tokenizer if self.tokenizer is not None else self.hparams.tokenizer,
                max_seq_len=self.hparams.max_seq_len,
                prompt_template=self.prompt_template,
                dataset_name=self.hparams.subset_name
                if self.hparams.subset_name is not None and self.hparams.subset_name != ""
                else self.hparams.dataset_name,
            )
        elif self.hparams.dataset_name in ["lambada", "lambada_zh"]:
            self.processor = NextWordDataProcessor(
                tokenizer=self.tokenizer if self.tokenizer is not None else self.hparams.tokenizer,
                max_seq_len=self.hparams.max_seq_len,
                prompt_template=self.prompt_template,
                dataset_name=self.hparams.subset_name
                if self.hparams.subset_name is not None and self.hparams.subset_name != ""
                else self.hparams.dataset_name,
            )
        elif self.hparams.subset_name in ["chid"]:
            self.processor = CtxPplDataProcessor(
                tokenizer=self.tokenizer if self.tokenizer is not None else self.hparams.tokenizer,
                max_seq_len=self.hparams.max_seq_len,
                prompt_template=self.prompt_template,
                dataset_name=self.hparams.subset_name
                if self.hparams.subset_name is not None and self.hparams.subset_name != ""
                else self.hparams.dataset_name,
            )
        elif self.hparams.dataset_name in ["ptb"]:
            self.processor = LLMPplDataProcessor(
                tokenizer=self.tokenizer if self.tokenizer is not None else self.hparams.tokenizer,
                max_seq_len=self.hparams.max_seq_len,
                prompt_template=None,
                dataset_name=self.hparams.dataset_name,
            )

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
            # no_sharding=True,
        )
        if self.hparams.gpu_prefetch:
            loader = GPUPrefetcher(loader)
        return loader
