# -*- coding:utf-8 -*-
# Email:    jiangxubin@bytedance.com
# Created:  2023-02-27 20:28:55
# Modified: 2023-02-27 20:28:55
import torch

from .tokenization import BertTokenizer


class DebertaTokenizer:
    def __init__(self, vocab_file, **kwargs):
        # VOCAB_FILE 见 1.3
        self.max_length = kwargs.get("max_len", 512)
        self.mask_padding_with_zero = True
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.special_tokens_count = 2
        self.pad_token_segment_id = 0
        self.tokenizer = BertTokenizer(
            vocab_file,
            do_lower_case=True,
            tokenize_emoji=False,
            greedy_sharp=True,
            max_len=self.max_length,
        )
        self.pad_id = self.tokenizer.vocab["[PAD]"]

    def __call__(self, text):
        if isinstance(text, list):
            input_ids, attention_mask, token_type_ids = [], [], []
            for sub_text in text:
                inputs = self._tokenize(sub_text)
                input_ids.append(inputs["input_ids"])
                attention_mask.append(inputs["attention_mask"])
                token_type_ids.append(inputs["token_type_ids"])
            return {
                "input_ids": torch.cat(input_ids, axis=0),
                "attention_mask": torch.cat(attention_mask, axis=0),
                "token_type_ids": torch.cat(token_type_ids, axis=0),
            }
        return self._tokenize(text)

    def _tokenize(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_length - self.special_tokens_count:
            tokens = tokens[: (self.max_length - self.special_tokens_count)]

        tokens = [self.cls_token] + tokens + [self.sep_token]
        token_type_ids = [0] * len(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1 if self.mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = self.max_length - len(input_ids)
        input_ids += [self.pad_id] * padding_length
        attention_mask += [0 if self.mask_padding_with_zero else 1] * padding_length
        token_type_ids += [self.pad_token_segment_id] * padding_length
        assert len(input_ids) == self.max_length
        assert len(attention_mask) == self.max_length
        assert len(token_type_ids) == self.max_length

        features = {
            "input_ids": torch.tensor(input_ids, dtype=torch.int32),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.int32),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.int32),
        }
        return features
