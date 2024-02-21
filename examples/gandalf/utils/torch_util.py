import io
import itertools as it
import sys
from collections import abc
from typing import List, Mapping

import numpy as np
import torch
import torch.autograd as autograd
import torch.distributed as dist
from packaging import version
from prettytable import PrettyTable
from utils.driver import DIST_CONTEXT, get_logger
from utils.file_util import hopen


def check_fft_version():
    # Acquires and parses the PyTorch version
    if version.parse(torch.__version__) >= version.parse("1.7"):
        if "torch.fft" not in sys.modules:
            raise RuntimeError("torch.fft module available but not imported")


def rfft(input_tensor, signal_ndim=1, n=None, dim=-1, norm=None) -> torch.Tensor:
    check_fft_version()
    if "torch.fft" not in sys.modules:
        return torch.rfft(input_tensor, signal_ndim=signal_ndim)
    else:
        return torch.fft.rfft(input_tensor, n, dim, norm)


def irfft(input_tensor, s=None, signal_ndim=1, dim=None, norm=None) -> torch.Tensor:
    check_fft_version()
    if "torch.fft" not in sys.modules:
        return torch.irfft(input_tensor, signal_ndim=signal_ndim, signal_sizes=s)
    else:
        return torch.fft.irfftn(input_tensor, s, dim, norm)


def collate_fields(data_list: List[Mapping[str, torch.Tensor]]):
    """collate result accordding to map keys"""
    results = []
    sample_dict = data_list[0]  # determine the schema by first data
    fields = list(sample_dict.keys())
    for key in fields:
        if isinstance(sample_dict[key], torch.Tensor):
            agg_data = torch.cat([data[key] for data in data_list], dim=0)
        else:
            agg_data = list(it.chain(*[data[key] for data in data_list]))
        results.append(agg_data)
    return dict(zip(fields, results))


def default_collate(batch, transposed_list=True):
    """puts each data field into a tensor with outer dimension batch size"""
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == "numpy" and elem_type.__name__ != "str_" and elem_type.__name__ != "string_":
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            return default_collate([torch.as_tensor(b) for b in batch], transposed_list)
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, (str, bytes)):
        return batch
    elif isinstance(elem, abc.Mapping):
        return {key: default_collate([d[key] for d in batch], transposed_list) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(default_collate(samples, transposed_list) for samples in zip(*batch)))
    elif isinstance(elem, abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if transposed_list and not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch) if transposed_list else batch
        return [default_collate(samples, transposed_list) for samples in transposed]


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    if isinstance(data, abc.Sequence) and not isinstance(data, str):
        return torch.tensor(data)
    if isinstance(data, int):
        return torch.LongTensor([data])
    if isinstance(data, float):
        return torch.FloatTensor([data])
    raise TypeError(f"type {type(data)} cannot be converted to tensor.")


def torch_io_load(filepath: str, **kwargs):
    """load model"""
    if not filepath.startswith("hdfs://"):
        return torch.load(filepath, **kwargs)
    with hopen(filepath, "rb") as reader:
        accessor = io.BytesIO(reader.read())
        state_dict = torch.load(accessor, **kwargs)
        del accessor
        return state_dict


def torch_io_save(obj, filepath: str, **kwargs):
    """save model"""
    if filepath.startswith("hdfs://"):
        with hopen(filepath, "wb") as writer:
            torch.save(obj, writer, **kwargs)
    else:
        torch.save(obj, filepath, **kwargs)


def conv_resblocks_state_to_ptx(state_dict):
    for k in list(state_dict.keys()):
        if k.endswith(".attn.in_proj_weight"):
            proj_q_w, proj_k_w, proj_v_w = state_dict[k].split(state_dict[k].size(0) // 3)
            del state_dict[k]
            state_dict[k.replace(".attn.in_proj_weight", ".attn.proj_q.weight")] = proj_q_w
            state_dict[k.replace(".attn.in_proj_weight", ".attn.proj_k.weight")] = proj_k_w
            state_dict[k.replace(".attn.in_proj_weight", ".attn.proj_v.weight")] = proj_v_w
        elif k.endswith(".attn.in_proj_bias"):
            proj_q_b, proj_k_b, proj_v_b = state_dict[k].split(state_dict[k].size(0) // 3)
            del state_dict[k]
            state_dict[k.replace(".attn.in_proj_bias", ".attn.proj_q.bias")] = proj_q_b
            state_dict[k.replace(".attn.in_proj_bias", ".attn.proj_k.bias")] = proj_k_b
            state_dict[k.replace(".attn.in_proj_bias", ".attn.proj_v.bias")] = proj_v_b
        elif k.endswith(".attn.out_proj.weight"):
            state_dict[k.replace(".attn.out_proj.weight", ".proj.weight")] = state_dict.pop(k)
        elif k.endswith(".attn.out_proj.bias"):
            state_dict[k.replace(".attn.out_proj.bias", ".proj.bias")] = state_dict.pop(k)
        elif k.endswith(".ln_1.weight"):
            state_dict[k.replace(".ln_1.weight", ".norm1.weight")] = state_dict.pop(k)
        elif k.endswith(".ln_1.bias"):
            state_dict[k.replace(".ln_1.bias", ".norm1.bias")] = state_dict.pop(k)
        elif k.endswith(".ln_2.weight"):
            state_dict[k.replace(".ln_2.weight", ".norm2.weight")] = state_dict.pop(k)
        elif k.endswith(".ln_2.bias"):
            state_dict[k.replace(".ln_2.bias", ".norm2.bias")] = state_dict.pop(k)
        elif k.endswith(".mlp.c_fc.weight"):
            state_dict[k.replace(".mlp.c_fc.weight", ".pwff.fc1.weight")] = state_dict.pop(k)
        elif k.endswith(".mlp.c_fc.bias"):
            state_dict[k.replace(".mlp.c_fc.bias", ".pwff.fc1.bias")] = state_dict.pop(k)
        elif k.endswith(".mlp.c_proj.weight"):
            state_dict[k.replace(".mlp.c_proj.weight", ".pwff.fc2.weight")] = state_dict.pop(k)
        elif k.endswith(".mlp.c_proj.bias"):
            state_dict[k.replace(".mlp.c_proj.bias", ".pwff.fc2.bias")] = state_dict.pop(k)
    return state_dict


def conv_resblocks_state_from_ptx(state_dict):
    for k in list(state_dict.keys()):
        if ".resblocks." not in k:
            continue
        if k.endswith(".attn.proj_q.weight"):
            proj_q_w = state_dict.pop(k)
            proj_k_w = state_dict.pop(k.replace(".attn.proj_q.weight", ".attn.proj_k.weight"))
            proj_v_w = state_dict.pop(k.replace(".attn.proj_q.weight", ".attn.proj_v.weight"))
            state_dict[k.replace(".attn.proj_q.weight", ".attn.in_proj_weight")] = torch.cat(
                [
                    proj_q_w,
                    proj_k_w,
                    proj_v_w,
                ]
            )
        elif k.endswith(".attn.proj_q.bias"):
            proj_q_b = state_dict.pop(k)
            proj_k_b = state_dict.pop(k.replace(".attn.proj_q.bias", ".attn.proj_k.bias"))
            proj_v_b = state_dict.pop(k.replace(".attn.proj_q.bias", ".attn.proj_v.bias"))
            state_dict[k.replace(".attn.proj_q.bias", ".attn.in_proj_bias")] = torch.cat(
                [
                    proj_q_b,
                    proj_k_b,
                    proj_v_b,
                ]
            )
        elif k.endswith(".proj.weight"):
            state_dict[k.replace(".proj.weight", ".attn.out_proj.weight")] = state_dict.pop(k)
        elif k.endswith(".proj.bias"):
            state_dict[k.replace(".proj.bias", ".attn.out_proj.bias")] = state_dict.pop(k)
        elif k.endswith(".norm1.weight"):
            state_dict[k.replace(".norm1.weight", ".ln_1.weight")] = state_dict.pop(k)
        elif k.endswith(".norm1.bias"):
            state_dict[k.replace(".norm1.bias", ".ln_1.bias")] = state_dict.pop(k)
        elif k.endswith(".norm2.weight"):
            state_dict[k.replace(".norm2.weight", ".ln_2.weight")] = state_dict.pop(k)
        elif k.endswith(".norm2.bias"):
            state_dict[k.replace(".norm2.bias", ".ln_2.bias")] = state_dict.pop(k)
        elif k.endswith(".pwff.fc1.weight"):
            state_dict[k.replace(".pwff.fc1.weight", ".mlp.c_fc.weight")] = state_dict.pop(k)
        elif k.endswith(".pwff.fc1.bias"):
            state_dict[k.replace(".pwff.fc1.bias", ".mlp.c_fc.bias")] = state_dict.pop(k)
        elif k.endswith(".pwff.fc2.weight"):
            state_dict[k.replace(".pwff.fc2.weight", ".mlp.c_proj.weight")] = state_dict.pop(k)
        elif k.endswith(".pwff.fc2.bias"):
            state_dict[k.replace(".pwff.fc2.bias", ".mlp.c_proj.bias")] = state_dict.pop(k)
    return state_dict


def smart_load_pretrained_state_dict(model, state_dict, show_load_status=True):
    state_dict = conv_resblocks_state_from_ptx(state_dict)
    parsed_state_dict = {}
    non_match_keys = []
    pretrained_keys = []
    for k, v in state_dict.items():
        if k in model.state_dict() or "amax" in k:
            parsed_state_dict[k] = v
            pretrained_keys.append(k)
        else:
            non_match_keys.append(k)

    if show_load_status and DIST_CONTEXT.is_local_master:
        table = PrettyTable(["Layer Name", "Weight Shape", "Data Type", "Load Success"])
        for k, v in model.named_parameters():
            table.add_row([k, v.shape, v.dtype, str(k in pretrained_keys)])
        table.align = "l"
        get_logger().info("\n###### Parameters ######\n{}".format(table.get_string()))
        get_logger().info("\n###### Not matched keys ######\n{}".format("\n".join(non_match_keys) + "\n"))
    DIST_CONTEXT.barrier()
    new_state_dict = model.state_dict()
    new_state_dict.update(parsed_state_dict)
    model.load_state_dict(new_state_dict)


def smarter_load_state_dict(model, pretrain_paths, prefix_changes=None, show_load_status=True):
    """Load Process: read hdfs/local pth file => parse => state_dict_load => print out"""
    if not pretrain_paths:
        get_logger().info("No pretrain paths found.")
        return

    if prefix_changes is None:
        prefix_changes = []

    resume_info = []
    # partial load pretrain state dict
    pretrain_state_dict_parsed = {}
    for i, pretrain_path in enumerate(pretrain_paths):
        if pretrain_path != "":
            get_logger().info("loading from pretrain path %s: %s " % (i, pretrain_path))
            pretrain_state_dict = torch_io_load(pretrain_path, map_location=lambda storage, loc: storage)
            resume_info.append(pretrain_state_dict)

            # extract model state dict
            if "state_dict" in pretrain_state_dict:
                pretrain_state_dict = pretrain_state_dict["state_dict"]
            elif "model" in pretrain_state_dict:
                pretrain_state_dict = pretrain_state_dict["model"]

            prefix_change = [prefix_change.split("->") for prefix_change in prefix_changes]
            for k, v in pretrain_state_dict.items():
                if k.startswith("module."):
                    k = k.replace("module.", "")
                no_match = True
                for pretrain_prefix, new_prefix in prefix_change:
                    if k.startswith(pretrain_prefix):
                        k = new_prefix + k[len(pretrain_prefix) :]
                        pretrain_state_dict_parsed[k] = v
                        no_match = False
                        break
                if no_match:
                    pretrain_state_dict_parsed[k] = v

    # pass to smart load
    smart_load_pretrained_state_dict(model, pretrain_state_dict_parsed, show_load_status=show_load_status)
    return resume_info


class AllGather(torch.autograd.Function):
    """An autograd function that performs all gather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, rank, world_size):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None,
            None,
        )


class GatherLayer(autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]
