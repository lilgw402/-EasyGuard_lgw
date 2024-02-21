# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Base classes common to both the slow and the fast tokenization classes: PreTrainedTokenizerBase (host all the user
fronting encoding methods) Special token mixing (host the special tokens logic) and BatchEncoding (wrap the dictionary
of output with special method for the Fast tokenizers)
"""
import io
from abc import ABC, abstractmethod
from collections import OrderedDict, UserDict
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import torch

from .. import __version__
from ..utils import hexists, hopen, logging, typecheck
from .hub import AutoHubClass

logger = logging.get_logger(__name__)


# TODO (junwei.Dong): 一些通用的方法注册到该基类中去，例如通过hdfs来load vocab文件，这时候这个通用方法一旦注册进基类中，那么所有子类获取hdfs的方法就可以调用基类的load
# EasyGuard tokenizer base class
class TokenizerBase(ABC, AutoHubClass):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def __call__(
        self,
        input_text: Union[str, List[str]],
        max_length: Optional[int] = None,
        text_completion: Optional[List[str]] = ["[CLS]", "[SEP]"],
        padding: Optional[str] = "[PAD]",
        input_segment_number: Optional[int] = 0,
        *args: Any,
        **kwds: Any,
    ) -> Union[OrderedDict, List[OrderedDict]]:
        """
        used to encode a sequence or a list of sequences

        Parameters
        ----------
        input_text : Union[str, List[str]]
            the input text str or list
        max_length : Optional[int], optional
            max length of `input_text`, by default None
        text_completion : Optional[List[str]], optional
            used to wrap the `input_text`, by default ["[CLS]", "[SEP]"]
        padding : Optional[str], optional
            when max_length > the length of `input_text`, use `padding` to fill in, by default "[PAD]"
        input_segment_number : Optional[int], optional
            the segment number, by default 0

        Returns
        -------
        Union[OrderedDict, List[OrderedDict]]
            the encoded sequence(s)

        Raises
        ------
        ValueError
            _description_
        """
        if not isinstance(input_text, (str, list)):
            raise ValueError(f"the type of argument `input_text` must be {str} or {list}")

        @typecheck(str)
        def _str_encode(text: str):
            text_tokens = self.tokenize(text)
            text_tokens_complete = [text_completion[0]] + text_tokens + [text_completion[-1]]
            token_ids = self.convert_tokens_to_ids(text_tokens_complete)
            padding_id = self.convert_tokens_to_ids([padding])
            extra_length = (max_length - len(token_ids)) if max_length else 0
            input_ids = token_ids[:max_length] + padding_id * extra_length
            input_mask = [1] * len(token_ids[:max_length]) + [0] * extra_length
            input_segment_ids = [input_segment_number] * len(input_ids)

            input_ids, input_mask, input_segment_ids = torch.tensor([[input_ids], [input_mask], [input_segment_ids]])
            return OrderedDict(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=input_segment_ids,
            )

        if isinstance(input_text, str):
            return _str_encode(input_text)
        else:
            return list(map(_str_encode, input_text))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        ...

    @classmethod
    def get_vocab(cls, vocab_file: str) -> str:
        # TODO (junwei.Dong): hget the vocab file and return the local path
        """if the vocab file in the remote server, get the file from server and return the local path

        Parameters
        ----------
        vocab_file : str
            path, such as: hdfs://xxx/vocab.txt

        Returns
        -------
        str
            the local path of the vocab file

        Raises
        ------
        ValueError
            _description_
        NotImplementedError
            _description_
        NotImplementedError
            _description_
        NotImplementedError
            _description_
        """
        ...

    @classmethod
    def load_vocab(cls, vocab_file: str) -> OrderedDict:
        """Loads a vocabulary file into a dictionary."""

        vocab = OrderedDict()
        index = 0
        if vocab_file.startswith("hdfs://"):
            if not hexists(vocab_file):
                raise ValueError("Can't find a vocabulary file at path '{}'. ".format(vocab_file))
            with hopen(vocab_file, "r") as reader:
                accessor = io.BytesIO(reader.read())
                while True:
                    token = accessor.readline()
                    token = token.decode("utf-8")  # 要解码使得数据接口类型一致
                    if not token:
                        break
                    token = token.strip()
                    vocab[token] = index
                    index += 1
                del accessor
                return vocab
        else:
            with io.open(vocab_file, "r", encoding="utf-8") as reader:
                while True:
                    token = reader.readline()
                    if not token:
                        break
                    token = token.strip()
                    vocab[token] = index
                    index += 1
                return vocab
        ...

    @abstractmethod
    def tokenize(self) -> List[str]:
        ...

    @abstractmethod
    def convert_tokens_to_ids(self) -> List[int]:
        ...

    @abstractmethod
    def convert_ids_to_tokens(self) -> List[str]:
        ...
