# -*- coding:utf-8 -*-
# Created By augustus at 2022-09-14 10:14:43
import json
import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from transformers import AutoModel, AutoTokenizer

config_file = "/mlx_devbox/users/jiangxubin/repo/EasyGuard/examples/gandalf/config/ecom_live_gandalf/live_gandalf_autodis_nn_asr.yaml"
traced_model_file = "/mlx_devbox/users/jiangxubin/repo/121/ecom_live_gandalf/ecom_live_gandalf_autodis_nn_asr_output_v4_1_202211031116/trace_model_fc_ts/traced_model.pt"
traced_preprocessor_model_file = "/mlx_devbox/users/jiangxubin/repo/121/maxwell/rh2_deploy/custom_ops/torchscript/ecom_live_gandalf/auto_dis_preprocess.jit"

with open(config_file, "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
feature_num = config["data"]["feature_provider"]["feature_num"]
author_embedding_dim = config["data"]["feature_provider"]["embedding_conf"]["author_embedding"]
asr_max_length = config["data"]["feature_provider"]["max_length"]
slot_mask = config["data"]["feature_provider"]["slot_mask"]
active_slot = [i for i in range(feature_num) if i not in slot_mask]
min_max_mapping = config["data"]["feature_provider"]["feature_norm_info"]


def process_text(text, use_random=False):
    if not use_random:
        if "deberta" in asr_encoder_config:
            asr_tokenizer = PtxDebertaTokenizer(asr_encoder_config)
            asr_inputs = asr_tokenizer(text, max_length=asr_max_length, return_tensors=True)

        else:
            asr_tokenizer = AutoTokenizer.from_pretrained(asr_encoder_config)
            asr_inputs = asr_tokenizer(
                text,
                max_length=asr_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        asr_inputs["input_ids"] = asr_inputs["input_ids"].cuda()
        asr_inputs["attention_mask"] = asr_inputs["attention_mask"].cuda()
        asr_inputs["token_type_ids"] = asr_inputs["token_type_ids"].cuda()
    else:
        asr_inputs = {
            "input_ids": torch.tensor([0.5 for i in range(asr_max_length)], dtype=torch.int32).reshape(1, -1).cuda(),
            "attention_mask": torch.tensor([0.5 for i in range(asr_max_length)], dtype=torch.int32)
            .reshape(1, -1)
            .cuda(),
            "token_type_ids": torch.tensor([0.5 for i in range(asr_max_length)], dtype=torch.int32)
            .reshape(1, -1)
            .cuda(),
        }
    return (
        asr_inputs["input_ids"],
        asr_inputs["attention_mask"],
        asr_inputs["token_type_ids"],
    )


def local_preprocessor_inference(traced_preprocessor_file):
    features = [idx * 0.01 for idx in range(feature_num)]
    features = torch.tensor(features, dtype=torch.float32).cuda().reshape(1, -1)
    # 加载preprocess.py打包的jit
    preprocess_module = torch.jit.load(traced_preprocessor_file).eval().cuda()
    auto_dis_input, feature_dense = preprocess_module(features)
    print(auto_dis_input)  # 校对预处理
    print(feature_dense)  # 校对预处理
    return auto_dis_input, feature_dense


def local_model_jit_inference(traced_model_file):
    feature_input_num = feature_num - len(slot_mask)

    # 训练过程中的预处理
    def process_feature_dense(features):
        feature_norm_info = min_max_mapping
        auto_dis_input = torch.zeros(feature_input_num, 3)
        feature_dense = torch.zeros(feature_input_num)
        slot_id = 0
        used_slot_id = 0
        for value in features:
            if slot_id not in slot_mask and slot_id < feature_num:
                # 特征归一化 norm_x = (x - min)/(max - min)
                norm_value = 0
                if value is None or np.isnan(value) or value < 0:  # 特征缺失或者为空, norm值为0
                    norm_value = 0
                elif value > feature_norm_info.get(str(used_slot_id), [0, 1])[1]:  # 大于最大值, norm值为1
                    norm_value = 1
                elif value < feature_norm_info.get(str(used_slot_id), [0, 1])[0]:  # 小于最大值, norm值为0
                    norm_value = 0
                else:  # 正常区间， norm = (x - min)/(max - min)
                    min_value = feature_norm_info.get(str(used_slot_id), [0, 1])[0]
                    max_value = feature_norm_info.get(str(used_slot_id), [0, 1])[1]
                    norm_value = (value - min_value) / (max_value - min_value)
                # x, x^2, sqrt(x)
                feature_dense[used_slot_id] = float(norm_value)  # dense layer
                auto_dis_input[used_slot_id][0] = float(norm_value)  # x
                auto_dis_input[used_slot_id][1] = float(norm_value * norm_value)  # x^2
                auto_dis_input[used_slot_id][2] = float(math.sqrt(norm_value))  # sqrt(x)
                used_slot_id += 1
            slot_id += 1
        return auto_dis_input, feature_dense

    # 去测试集或者线上捞一个case
    # 策略数值特征
    features = [0.5] * feature_num
    auto_dis_input, feature_dense = process_feature_dense(features)
    auto_dis_input = torch.tensor(auto_dis_input, dtype=torch.float32).cuda().unsqueeze(-3)
    feature_dense = torch.tensor(feature_dense, dtype=torch.float32).cuda().reshape(1, -1)
    print("auto_dis_input: ", auto_dis_input)  # 校对预处理
    print("feature_dense: ", feature_dense)  # 校对预处理
    # 达人embedding特征
    author_embedding = [0.5] * author_embedding_dim
    author_embedding = torch.tensor(author_embedding, dtype=torch.float32).cuda().reshape(1, -1)
    # asr embedding特征
    input_ids, attention_mask, token_type_ids = process_text("家人们这个款式的衣服只要九十八米啊走过路过不要错过")
    # 导入模型到GPU中
    model = torch.jit.load(traced_model_file).eval().cuda()
    prob = model(
        auto_dis_input,
        feature_dense,
        author_embedding,
        input_ids,
        attention_mask,
        token_type_ids,
    )
    print(prob)


def local_preprocessor_local_model_jit_inference(traced_preprocessor_model_file, traced_model_file):
    features = [0.5] * feature_num
    features = torch.tensor(features, dtype=torch.float32).cuda().reshape(1, -1)
    # 加载preprocess.py打包的jit
    preprocess_module = torch.jit.load(traced_preprocessor_model_file).eval().cuda()
    auto_dis_input, feature_dense = preprocess_module(features)
    print(auto_dis_input)  # 校对预处理
    print(feature_dense)  # 校对预处理
    # 达人embedding特征
    author_embedding = [0.5] * author_embedding_dim
    author_embedding = torch.tensor(author_embedding, dtype=torch.float32).cuda().reshape(1, -1)
    # asr embedding特征
    input_ids, attention_mask, token_type_ids = process_text("家人们这个款式的衣服只要九十八米啊走过路过不要错过")
    # 加载本地的模型
    model = torch.jit.load(traced_model_file).eval().cuda()
    prob = model(
        auto_dis_input,
        feature_dense,
        author_embedding,
        input_ids,
        attention_mask,
        token_type_ids,
    )
    print(prob)


if __name__ == "__main__":
    text = "家人们这个款式的衣服只要九十八米啊走过路过不要错过"
    features = [0.5 for idx in range(feature_num + 40)]
    print(len(features))
    # Local preprocessor jit inference
    auto_dis_input, feature_dense = local_preprocessor_inference(traced_preprocessor_model_file, features)
    # Local test model jit inference
    local_model_jit_inference(traced_model_file)
    # Local test autodis preprocessor and model jit inference
    local_preprocessor_local_model_jit_inference(traced_preprocessor_model_file, traced_model_file)
