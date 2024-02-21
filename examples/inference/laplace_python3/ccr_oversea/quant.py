# 对Llama2模型的weight做int8的量化

import torch


def quant_weight(weight):
    """对linear层的weight做int8的量化,默认weight是float16"""
    assert len(weight.shape) == 2, "make sure weight is 2D"
    max_v = torch.max(torch.abs(weight), dim=1)
    scale = max_v.values / 127
    # torch.round在CPU上不支持float16操作
    w_fp32 = torch.tensor(weight, dtype=torch.float32)
    scale_fp32 = torch.tensor(scale, dtype=torch.float32)
    q_v = torch.clip(torch.round(w_fp32 / scale_fp32.unsqueeze(1)).to(torch.float32), -127, 127)
    q_v = torch.tensor(q_v, dtype=torch.float16)
    return scale, q_v


def quant_linear(linear):
    with torch.no_grad():
        S, V = quant_weight(linear.weight)
        # 满足xperf可以接受的格式
        linear.weight = torch.nn.Parameter(V, requires_grad=False)
        linear.weight_qscale = torch.nn.Parameter(S, requires_grad=False)
        del S
        del V


def quant_model(model):
    for block in model.model.layers:
        # 对block里面的每个linear层做量化
        quant_linear(block.self_attn.q_proj)
        quant_linear(block.self_attn.k_proj)
        quant_linear(block.self_attn.v_proj)
        quant_linear(block.self_attn.o_proj)
        quant_linear(block.mlp.gate_proj)
        quant_linear(block.mlp.up_proj)
        quant_linear(block.mlp.down_proj)


if __name__ == "__main__":
    from .handler_hf import EndpointHandler

    eh = EndpointHandler()
    model = eh.model
    quant_model(model=model)
