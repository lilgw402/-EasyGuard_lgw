"""
author:        jiangxubin <jiangxubin@bytedance.com>
lastModified:  2023-03-13 03:49:00
"""
# -*- coding:utf-8 -*-
# Created By augustus at 2022-09-14 10:14:43


from typing import Dict, List

import torch
import yaml

# from transformers import AutoTokenizer, AutoModel

config_file = "/mlx_devbox/users/jiangxubin/repo/EasyGuard/examples/gandalf/config/ecom_live_gandalf/live_gandalf_autodis_nn_asr.yaml"  # noqa: E501
traced_model_file = "/mlx_devbox/users/jiangxubin/repo/121/ecom_live_gandalf/ecom_live_gandalf_autodis_nn_asr_output_v4_1_202211031116/trace_model_fc_ts/traced_model.pt"  # noqa: E501
traced_preprocessor_model_file = "/mlx_devbox/users/jiangxubin/repo/121/maxwell/rh2_deploy/custom_ops/torchscript/ecom_live_gandalf/auto_dis_preprocess.jit"  # noqa: E501

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


class AutoDisPreprocessor(torch.nn.Module):
    def __init__(
        self,
        feature_num: int,
        active_slot: List[int],
        min_max_mapping: Dict[str, List[float]],
    ):
        super(AutoDisPreprocessor, self).__init__()

        active_slot = sorted(active_slot)
        compressed_min = [min_max_mapping.get(str(slot_id), [0, 1])[0] for slot_id in active_slot]
        compressed_max = [min_max_mapping.get(str(slot_id), [0, 1])[1] for slot_id in active_slot]

        self.num_active_slot = len(active_slot)
        self.active_slot = torch.tensor(active_slot, dtype=torch.long).reshape(self.num_active_slot).cuda()
        self.compressed_min = torch.tensor(compressed_min, dtype=torch.float32).reshape(1, self.num_active_slot).cuda()
        self.compressed_max = torch.tensor(compressed_max, dtype=torch.float32).reshape(1, self.num_active_slot).cuda()
        self.compressed_range = self.compressed_max - self.compressed_min

    def forward(self, features):  # 仅处理离散特征，embedding/直接输入，不需要预处理，cv/nlp在其他地方预处理
        # features shape: [Batch, NumFeatures]
        feature_dense = features[:, self.active_slot].contiguous()
        feature_dense_norm = (feature_dense - self.compressed_min) / self.compressed_range
        feature_dense_norm = torch.clamp(feature_dense_norm, min=0.0, max=1.0)
        # 特征缺失或者为空(规范为-1), norm值为0
        feature_dense_norm[feature_dense == -1] = 0.0
        feature_dense_norm[feature_dense.isnan()] = 0.0

        auto_dis_input_list = [
            feature_dense_norm,
            feature_dense_norm * feature_dense_norm,
            torch.sqrt(feature_dense_norm),
        ]
        auto_dis_input = torch.stack(auto_dis_input_list, dim=2)
        return auto_dis_input, feature_dense_norm


def trace_autodis_preprocessor(traced_preprocessor_model_file):
    module = AutoDisPreprocessor(
        feature_num=feature_num,
        active_slot=active_slot,
        min_max_mapping=min_max_mapping,
    ).cuda()
    feature_tensor = torch.tensor([0.5 for i in range(feature_num)], dtype=torch.float32).reshape(1, -1).cuda()
    sample_input_tensor = feature_tensor
    jit_module = torch.jit.trace(module, sample_input_tensor).cuda()
    jit_module.save(traced_preprocessor_model_file)


if __name__ == "__main__":
    # Jit trace autodis preprocessor
    trace_autodis_preprocessor(traced_preprocessor_model_file)
