# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass, field

import torch
from models.deberta4classification import DebertaClassifier
from ptx.amp import amp_convert_network

base_dir_path = "/mlx_devbox/users/wanli.0815/repo/matx_model_exporter/resources/ccr_clothes_industry"


@dataclass
class TraceConfig(object):
    model_name_or_path: str = field(default=base_dir_path)
    pt_file_path: str = field(default=os.path.join(base_dir_path, "ptx_model.pt"))
    output_path: str = field(default=base_dir_path)
    num_classes: int = field(default=42)


trace_configs = TraceConfig()


def trace_model():
    assert trace_configs.pt_file_path
    model = DebertaClassifier(
        config=trace_configs.model_name_or_path,
        num_labels=trace_configs.num_classes,
    )

    model = torch.nn.DataParallel(model)
    model = model.module
    model.load_state_dict(torch.load(trace_configs.pt_file_path))
    print(f"Successfully loaded pretrained model from: {trace_configs.pt_file_path}")

    model = amp_convert_network(model, torch.float16, hack_forward=True)
    device = torch.device("cuda:0")
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    input_ids = torch.randint(1, 145607, (1, 512)).long()
    segment_ids = torch.zeros((1, 512)).long()
    attention_mask = torch.ones_like(input_ids)
    traced_model = torch.jit.trace(
        model,
        (
            input_ids.to(device),
            segment_ids.to(device),
            attention_mask.to(device),
        ),
    )
    torch.jit.save(traced_model, os.path.join(trace_configs.output_path, "model.jit"))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    trace_model()

    # Validation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jit_model = torch.jit.load(
        os.path.join(trace_configs.output_path, "model.jit"),
        map_location=device,
    )
    input_ids = torch.randint(1, 145607, (1, 512)).long().to(device)
    segment_ids = torch.zeros((1, 512)).long().to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    model_ret = jit_model(input_ids, segment_ids, attention_mask)
    assert tuple(model_ret.shape) == (1, trace_configs.num_classes)
