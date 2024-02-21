# -*- coding: utf-8 -*-

import argparse

import torch
from titan import TOSHelper, create_model, list_models


def parse_args():
    # Parse args with argparse tool
    parser = argparse.ArgumentParser(description="Load Model from Model Zoo")
    parser.add_argument("--tos_bucket", type=str, help="model zoo bucket name")
    parser.add_argument("--tos_access_key", type=str, help="model zoo bucket access key")
    parser.add_argument("--model_name", type=str, help="model name")
    parser.add_argument("--model_version", type=str, help="model version")
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="model input image size, default is 224",
    )
    parser.add_argument(
        "--features_only",
        action="store_true",
        help="whether to only output features",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="whether to use pretrained weights, default is False.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    tos_helper = TOSHelper(args.tos_bucket, args.tos_access_key)

    # list models
    print("Titan models: ")
    for key in list_models():
        print("\t" + key)

    # 创建模型
    model = create_model(
        model_name=args.model_name,
        pretrained=args.pretrained,
        pretrained_version=args.model_version,
        features_only=args.features_only,
        tos_helper=tos_helper,
    )

    # 准备数据，实际使用时每次可能会从dataloader中取出预处理好的数据送入网络。这里使用随机数据作为样例
    data_batch = torch.rand(2, 3, args.img_size, args.img_size)

    # 网络向前传播并产生特征
    out = model(data_batch)
    print("outputs\n", out)


# Outputs:
#   torch.Size([2, 256, 56, 56])
#   torch.Size([2, 512, 28, 28])
#   torch.Size([2, 1024, 14, 14])
#   torch.Size([2, 2048, 7, 7])

if __name__ == "__main__":
    main()
