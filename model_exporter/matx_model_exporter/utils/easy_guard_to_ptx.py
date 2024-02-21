# -*- coding: utf-8 -*-
from collections import OrderedDict

import torch


def convert(input_file_path: str, output_file_path: str) -> None:
    model_weights = torch.load(input_file_path)
    new_weights = OrderedDict()
    for name, value in model_weights.items():
        name = name.split(".", 1)[-1] if "deberta" in name else name
        new_weights[name] = value
    for name, value in new_weights.items():
        print(f"Name: {name: >45s}, shape: {str(value.shape): >30s}")
    torch.save(new_weights, output_file_path)


if __name__ == "__main__":
    input_file_path = "/mlx_devbox/users/wanli.0815/repo/matx_model_exporter/resources/ccr_clothes_industry/model.pt"
    out_file_path = "/mlx_devbox/users/wanli.0815/repo/matx_model_exporter/resources/ccr_clothes_industry/ptx_model.pt"
    convert(input_file_path=input_file_path, output_file_path=out_file_path)
