"""An customizable fashion_deberta example"""
import json
import math
import os
import random
import sys
from typing import List, Union

import numpy as np
import torch

try:
    import easyguard
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from cruise.data_module import CruiseDataModule
from cruise.data_module.cruise_loader import DistributedCruiseDataLoader
from cruise.utilities.hdfs_io import hlist_files, hopen

from easyguard.core import AutoTokenizer
from easyguard.utils.data_helpers import build_vocab


class TextProcessor:
    def __init__(
        self,
        text_label_field,
        text_field,
        tokenizer,
        context_length,
        vocab,
        classification_task_enable,
        cl_enable,
        span_set,
        max_forward_search_step,
        multi_label_enable,
        multi_label_map,
        multi_label_split_token,
        mlm_probability,
    ):
        self._text_label_field = text_label_field
        self._text_field = text_field
        self._tokenizer = tokenizer
        self._context_length = context_length
        self._mlm_probability = mlm_probability
        self._vocab = vocab
        self._classification_task_enable = classification_task_enable
        self._cl_enable = cl_enable
        self._span_set = span_set
        self._multi_label_enable = multi_label_enable
        self._multi_label_map = multi_label_map
        self._multi_label_split_token = multi_label_split_token
        self.PAD_IDX = self._vocab["[PAD]"]
        self.SEP_IDX = self._vocab["[SEP]"]
        self.CLS_IDX = self._vocab["[CLS]"]
        self.MASK_IDX = self._vocab["[MASK]"]
        # 遍历tokens列表，每个位置向前看3个token，去trie树检索，最后取最长的span进行mask
        self._max_forward_search_step = max_forward_search_step
        self.cnt = 0

    def transform(self, data_dict: dict):
        # get image by key, order matters
        if not self._text_field in data_dict:
            raise KeyError(f"Unable to find text by keys: {self._text_field} available keys: {data_dict.keys()}")
        text = data_dict.get(self._text_field, "")
        text_token = self._tokenizer.tokenize(text)
        # 对最后的tokenizer结果进行处理，避免分词策略影响效果和处理逻辑
        text_token = text_token[: self._context_length - 2]
        input_ids, mlm_input_ids, mlm_labels = self.mlm_span_mask(text_token)
        if self.cnt < 1:
            print(f"input_ids: {input_ids}")
            print(f"mlm_input_ids: {mlm_input_ids}")
            mlm_input_tokens = [self._vocab.itos[idx] for idx in mlm_input_ids]
            print(f"mlm_input_tokens: {mlm_input_tokens}")
            print(f"mlm_labels: {mlm_labels}")
            self.cnt += 1
        return_dict = {
            "input_ids": input_ids,
            "mlm_input_ids": mlm_input_ids,
            "mlm_labels": mlm_labels,
        }
        if self._classification_task_enable:
            if not self._text_label_field in data_dict:
                raise KeyError(
                    f"Unable to find text by keys: {self._text_label_field}, available keys: {data_dict.keys()}"
                )
            if self._multi_label_enable:
                # multi label
                multi_label = np.zeros(len(self._multi_label_map), dtype=int)
                for l in data_dict.get(self._text_label_field).split(self._multi_label_split_token):
                    # 去掉不在l2map的label
                    if l not in self._multi_label_map.keys():
                        print(f"[WARNING] label {l} not in multi_label_map!!!")
                        continue
                    multi_label[self._multi_label_map[l]] = 1
                return_dict["classification_labels"] = multi_label
            else:
                label = int(data_dict.get(self._text_label_field))
                # label = int(ccr_label_map[data_dict.get(self._text_label_field)])
                return_dict["classification_labels"] = int(label)
        return return_dict

    def batch_transform(self, batch_data):
        # batch_data: List[Dict[modal, modal_value]]
        input_ids = []
        mlm_input_ids = []
        input_mask = []
        input_segment_ids = []
        mlm_labels = []
        classification_labels = []
        max_len = self._context_length

        for ib, ibatch in enumerate(batch_data):
            mlm_input_ids.append(
                ibatch["mlm_input_ids"][:max_len] + [self.PAD_IDX] * (max_len - len(ibatch["mlm_input_ids"]))
            )
            input_ids.append(ibatch["input_ids"][:max_len] + [self.PAD_IDX] * (max_len - len(ibatch["input_ids"])))
            input_mask.append(
                [1] * len(ibatch["mlm_input_ids"][:max_len]) + [0] * (max_len - len(ibatch["mlm_input_ids"]))
            )
            input_segment_ids.append([0] * max_len)
            mlm_labels.append(ibatch["mlm_labels"][:max_len] + [-100] * (max_len - len(ibatch["mlm_labels"])))

            if self._classification_task_enable:
                classification_labels.append(ibatch["classification_labels"])

        # for cl, double data
        if self._cl_enable:
            input_ids.extend(input_ids)  # same data extended in the end
            input_mask.extend(input_mask)
            input_segment_ids.extend(input_segment_ids)
            if self._classification_task_enable:
                classification_labels.extend(classification_labels)

        mlm_labels = torch.tensor(mlm_labels)
        mlm_input_ids = torch.tensor(mlm_input_ids)
        input_ids = torch.tensor(input_ids)
        input_mask = torch.tensor(input_mask)
        input_segment_ids = torch.tensor(input_segment_ids)

        res = {
            "mlm_labels": mlm_labels,
            "mlm_input_ids": mlm_input_ids,
            "input_ids": input_ids,
            "input_masks": input_mask,
            "input_segment_ids": input_segment_ids,
        }
        if self._classification_task_enable:
            res["classification_labels"] = torch.tensor(np.array(classification_labels))
        return res

    def mlm_span_mask(self, input_tokens):
        """
        优先mask 词表中词，剩下的random mask
        """
        input_ids = []
        mlm_input_ids = []
        mlm_labels = []

        def get_mask_replace_token(token_id):
            """
            80%的概率直接替换成Mask；
            10%的概率替换其它词；
            10%的概率不进行替换；
            """
            prob = random.random()
            if 0 <= prob < 0.8:
                res = self.MASK_IDX
            elif 0.8 <= prob < 0.9:
                res = random.randint(0, len(self._vocab) - 1)
            else:
                res = token_id
            return res

        # 去掉## 再去检索，尽量不受分词影响
        text_token_rm_shape = [token.replace("##", "") for token in input_tokens]
        # 哪些位置进行span mask
        mlm_mask_idxs = [0] * len(text_token_rm_shape)

        # 先过一遍，如果存在关键span，则都mask上，采用最长匹配原则
        i = 0
        text_token_len = len(text_token_rm_shape)
        while i < text_token_len:
            max_span_len = -1
            span_str = ""
            for j in range(self._max_forward_search_step):
                if i + j + 1 > text_token_len:
                    break
                span_list = text_token_rm_shape[i : i + j + 1]
                span_str = "".join(span_list)
                if span_str in self._span_set:
                    max_span_len = max(j, max_span_len)
            if max_span_len > -1:
                for inx in range(i, i + max_span_len + 1):
                    mlm_mask_idxs[inx] = 1
            if max_span_len == 0 or max_span_len == -1:
                i = i + 1
            else:
                i = i + max_span_len + 1
            # if max_span_len > 3:
            #     print(f"max_span_len: {max_span_len}")
        # if sum(mlm_mask_idxs) > 0:
        #     print(f"text_tokens: {input_tokens}")
        #     print(f"mlm_mask_idxs: {mlm_mask_idxs}")
        # 如果span mask不够mlm_mask_num，剩下的用rand mask补齐
        total_mask_num = math.ceil(len(input_tokens) * self._mlm_probability)
        total_span_len = min(total_mask_num, sum(mlm_mask_idxs))
        total_rand_len = total_mask_num - total_span_len
        # print(f"total_span_len: {total_span_len}")
        # print(f"total_rand_len: {total_rand_len}")
        # handle [CLS]
        mlm_labels.append(-100)
        mlm_input_ids.append(self.CLS_IDX)
        input_ids.append(self.CLS_IDX)
        i = 0
        span_i = 0
        rand_i = 0
        while i < len(input_tokens):
            token_id = self._vocab[input_tokens[i]]
            # span mask handle
            if mlm_mask_idxs[i] == 1:
                if span_i < total_span_len:
                    mlm_labels.append(token_id)
                    mlm_input_ids.append(get_mask_replace_token(token_id))
                    span_i += 1
                # 超出mlm_prob限制，直接不mask了
                else:
                    mlm_labels.append(-100)
                    mlm_input_ids.append(token_id)
            else:
                # rand mask补齐了，剩下的rand位置直接不mask了
                if rand_i >= total_rand_len:
                    mlm_labels.append(-100)
                    mlm_input_ids.append(token_id)
                    rand_i += 1
                else:
                    prob = random.random()
                    # print(f"rand prob: {prob}")
                    if prob <= self._mlm_probability:
                        mlm_labels.append(token_id)
                        mlm_input_ids.append(get_mask_replace_token(token_id))
                        rand_i += 1
                    else:
                        mlm_labels.append(-100)
                        mlm_input_ids.append(token_id)
            input_ids.append(token_id)
            i += 1
            # handle [SEP]
        mlm_labels.append(-100)
        mlm_input_ids.append(self.SEP_IDX)
        input_ids.append(self.SEP_IDX)

        return input_ids, mlm_input_ids, mlm_labels

    def torch_mask_tokens(self, inputs, special_tokens_mask):
        """
        copyed from: huggingface
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self._mlm_probability)

        special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.MASK_IDX

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self._vocab), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class FashionDataModule(CruiseDataModule):
    def __init__(
        self,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        train_paths: Union[
            str, List[str]
        ] = "hdfs://haruna/home/byte_ecom_govern/user/yangzheming/chinese/common_model/trails/ccr_v3_live_0.3b_mlm/traindata/part*",
        val_paths: Union[
            str, List[str]
        ] = "hdfs://haruna/home/byte_ecom_govern/user/yangzheming/chinese/common_model/trails/ccr_v3_live_0.3b_mlm/validdata/part*",
        data_size: int = 1000,
        val_step: int = 10,
        classification_task_enable: bool = False,
        text_label_field: str = "label",
        text_field: str = "text",
        cl_enable: bool = True,
        num_workers: int = 1,
        context_length: int = 512,
        mlm_probability: float = 0.15,
        key_span_file_path: str = "",
        max_forward_search_step: int = 6,
        multi_label_enable: bool = False,
        multi_label_map_file: str = "",
        multi_label_split_token: str = "@##@",
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
        self.cl_enable = self.hparams.cl_enable

        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrain_model_name)
        self.vocab = build_vocab(self.hparams.vocab_file_path)
        # Entity Mask
        self.span_set = set()
        if self.hparams.key_span_file_path:
            with hopen(self.hparams.key_span_file_path, "r") as fr:
                for line in fr:
                    line = line.decode("utf-8")
                    line = line.strip()
                    if line:
                        self.span_set.add(line)
            print(f"span_set len: {len(self.span_set)}")
        self.max_forward_search_step = self.hparams.max_forward_search_step
        # Multi-Label
        self.multi_label_map = {}
        self.multi_label_split_token = ""
        if self.hparams.multi_label_enable:
            with hopen(self.hparams.multi_label_map_file) as reader:
                self.multi_label_map = json.loads(reader.read())
            self.multi_label_split_token = self.hparams.multi_label_split_token

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
                self.hparams.classification_task_enable,
                self.cl_enable,
                self.span_set,
                self.max_forward_search_step,
                self.hparams.multi_label_enable,
                self.multi_label_map,
                self.multi_label_split_token,
                mlm_probability=self.hparams.mlm_probability,
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
                self.hparams.classification_task_enable,
                self.cl_enable,
                self.span_set,
                self.max_forward_search_step,
                self.hparams.multi_label_enable,
                self.multi_label_map,
                self.multi_label_split_token,
                mlm_probability=self.hparams.mlm_probability,
            ),
            predefined_steps=self.hparams.val_step,
            source_types=["jsonl"],
            shuffle=False,
        )
