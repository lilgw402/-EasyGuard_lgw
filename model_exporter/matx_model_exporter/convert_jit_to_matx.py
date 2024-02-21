# -*- coding: utf-8 -*-

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import matx
import torch
from ptx.ops.fastertransformer.inference import FT_INFER_LIB_ROOT
from text_cutter import Cutter
from text_tokenizer import BpeTokenizer

ft_op_libpath = "libtorch_ops_dyn.so"
torch.ops.load_library(os.path.join(FT_INFER_LIB_ROOT, ft_op_libpath))  # load fastertransformer的op


@dataclass
class MatxConfigs(object):
    base_dir: str = field(
        default="/mlx_devbox/users/wanli.0815/repo/matx_model_exporter/models_output/ccr_clothes_industry"
    )
    libcut_path: str = field(default="/opt/tiger/libcut_data_zh_20200827fix2")
    service_name: str = field(default="matx_service_for_clothes")
    pad_token: str = field(default="[PAD]")
    unk_token: str = field(default="[UNK]")
    cls_token: str = field(default="[CLS]")
    sep_token: str = field(default="[SEP]")
    # vocab_path: str = field(default='Vocabulary file path')
    # jit_path: str = field(default='Jit file path')
    num_classes: int = field(default=42)
    max_len: int = field(default=512)
    device: int = field(default=0)


class Vocabulary(object):
    __slots__: Tuple[Dict] = ["vocab"]

    def __init__(self, vocab_file: str) -> None:
        self.vocab = matx.Dict()

        f = open(vocab_file)
        idx = 0
        for line in f:
            word = line.rstrip("\n")
            self.vocab[word] = idx
            idx += 1

    def lookup(self, key: str) -> int:
        if key in self.vocab:
            return self.vocab[key]
        else:
            return -1


class BertInputsBuilder(object):
    def __init__(
        self,
        vocab: Vocabulary,
        pad_token: str,
        unk_token: str,
        cls_token: str,
        sep_token: str,
        max_len: int,
    ) -> None:
        self.vocab: Vocabulary = vocab
        self.pad: int = vocab.lookup(pad_token)
        self.unk: int = vocab.lookup(unk_token)
        self.cls: int = vocab.lookup(cls_token)
        self.sep: int = vocab.lookup(sep_token)
        self.max_len: int = max_len

    def __call__(self, batch_token_list: List[List[str]]) -> Tuple[matx.NDArray, matx.NDArray, matx.NDArray]:
        shapes = matx.List()
        shapes.reserve(2)
        shapes.append(len(batch_token_list))
        shapes.append(self.max_len)
        input_ids = matx.List()
        input_ids.reserve(shapes[0] * shapes[1])
        attention_mask = matx.List()
        attention_mask.reserve(shapes[0] * shapes[1])
        segment_id = matx.List()
        segment_id.reserve(shapes[0] * shapes[1])

        for idx, token_list in enumerate(batch_token_list):
            cur_token_num = len(token_list)
            input_ids.append(self.cls)
            segment_id.append(0)
            attention_mask.append(1)
            for i in range(self.max_len - 1):
                j = -1
                temp = 1
                if i < cur_token_num:
                    if i == self.max_len - 2:
                        j = self.sep
                    else:
                        a = "".join(token_list[i])
                        j = self.vocab.lookup(a)
                        if j == -1:
                            j = self.unk
                else:
                    if i == cur_token_num:
                        j = self.sep
                    else:
                        j = self.pad
                        temp = 0
                attention_mask.append(temp)
                input_ids.append(j)
                segment_id.append(0)

        return (
            matx.NDArray(input_ids, shapes, "int64"),
            matx.NDArray(attention_mask, shapes, "int64"),
            matx.NDArray(segment_id, shapes, "int64"),
        )


def for_input(text: List[bytes], Cutter: Any, Tokenizer: Any) -> List[List[str]]:
    all_tokens = []
    max_len = 512
    for i in text:
        words = Cutter(i.decode(), "FINE")
        tokens = []
        for word in words:
            tokens.append(Tokenizer(word))
        if len(tokens) > max_len - 2:
            tokens = tokens[: max_len - 2]
        all_tokens.append(tokens)
    return all_tokens


def get_results(infer_rets: matx.NDArray) -> List[bytes]:
    ret = list()
    for infer_ret in infer_rets.tolist():
        ret.append(json.dumps({_: score for _, score in enumerate(infer_ret)}))
    return ret


class Pipeline(object):
    def __init__(self, configs: dataclass):
        self.configs = configs
        vocab_path = os.path.join(self.configs.base_dir, "vocab.txt")
        torch_model = os.path.join(self.configs.base_dir, "model.jit")
        self.segmentor = matx.script(Cutter)("CRF_LARGE", self.configs.libcut_path)

        self.tokenizer = matx.script(BpeTokenizer)(vocab_path, wordpiece_type="bert", lower_case=True)

        vocab = matx.script(Vocabulary)(vocab_path)
        self.input_builder = matx.script(BertInputsBuilder)(
            vocab=vocab,
            pad_token=self.configs.pad_token,
            unk_token=self.configs.unk_token,
            cls_token=self.configs.cls_token,
            sep_token=self.configs.sep_token,
            max_len=self.configs.max_len,
        )
        self.for_input = matx.script(for_input)
        self.torch_model = matx.script(
            torch.jit.load(
                torch_model,
                map_location=f"cuda:{self.configs.device}" if self.configs.device >= 0 else "cpu",
            )
        )
        self.get_results = matx.script(get_results)

    def preprocess(self, text: List[bytes]):
        all_tokens = self.for_input(text, self.segmentor, self.tokenizer)
        input_ids, input_mask, segment_ids = self.input_builder(all_tokens)
        return input_ids, input_mask, segment_ids

    def process(self, text: List[bytes]):
        input_ids, attention_mask_all, segment_id_all = self.preprocess(text)
        infer_rets = self.torch_model(input_ids, segment_id_all, attention_mask_all)
        output = self.get_results(infer_rets)
        return output


if __name__ == "__main__":
    matx_configs = MatxConfigs()
    ppl = Pipeline(configs=matx_configs)
    test_case = {"text": ["这个衣服掉色严重啊".encode()]}
    ret_before_trace = ppl.process(**test_case)

    save_path = os.path.join(matx_configs.base_dir, matx_configs.service_name)
    traced_model = matx.trace(ppl.process, **test_case)
    ret_after_trace = traced_model.run(test_case)
    for y_before_trace, y_after_trace in zip(ret_before_trace, ret_after_trace):
        assert y_before_trace == y_after_trace
    matx.save(traced_model, save_path)

    # # Reload
    loaded_model = matx.load(save_path, 0)  # On device 0
    ret_from_saved = loaded_model.run(test_case)
    for y_before_trace, y_from_saved in zip(ret_before_trace, ret_from_saved):
        assert y_before_trace == y_from_saved

    # Warmup Data
    allowed_batch_sizes = [1, 2, 4, 8]
    all_data = []
    for bs in allowed_batch_sizes:
        data = {"text": [test_case["text"][0]] * bs}
        all_data.append(data)

    with open(f"{save_path}/warmup_data.json", "w") as fw:
        fw.write(matx.serialize(all_data))

    # meta data
    with open(f"{save_path}/meta.pb.txt", "w") as fw:
        fw.write(f'name:"{matx_configs.service_name}"\n')
        for bs in allowed_batch_sizes:
            fw.write(f"allowed_batch_sizes:{bs}\n")
