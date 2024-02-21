import logging

import cruise as crs
import torch

try:
    import deepspeed
except ImportError:
    deepspeed = None


class TokenCheckpointHook:
    def on_load_checkpoint(
        self,
        trainer: "crs.CruiseTrainer",
        model: "crs.CruiseModule",
        *args,
        **kwargs,
    ) -> None:
        checkpoint = kwargs["checkpoint"]
        model.average_token_rate = checkpoint.get("average_token_rate", 0.95)
        model.consume_tokens = checkpoint.get("consume_tokens", 0)

    def on_save_checkpoint(
        self,
        trainer: "crs.CruiseTrainer",
        model: "crs.CruiseModule",
        *args,
        **kwargs,
    ) -> None:
        checkpoint = kwargs["checkpoint"]
        checkpoint["average_token_rate"] = model.average_token_rate
        checkpoint["consume_tokens"] = model.consume_tokens


def load_zero3_state_dict(state_dict, module, metadata, error_msgs, prefix="", verbose=False):
    if deepspeed is None:
        raise RuntimeError("deepspeed cannot be imported.")
    local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
    args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
    # because zero3 puts placeholders in model params, this context
    # manager gathers (unpartitions) the params of the current layer, then loads from
    # the state dict and then re-partitions them again
    with deepspeed.zero.GatheredParameters(list(module.parameters(recurse=False)), modifier_rank=0):
        if torch.distributed.get_rank() == 0:
            module._load_from_state_dict(*args)
    for name, child in module._modules.items():
        if child is not None:
            if verbose:
                logging.info(
                    f'Loading {prefix + name + "."}, param count: {sum(param.numel() for param in child.parameters())}'
                )
            load_zero3_state_dict(
                state_dict,
                child,
                metadata,
                error_msgs,
                prefix=prefix + name + ".",
                verbose=verbose,
            )


def is_zero3(cfg):
    try:
        level = cfg["trainer"]["accelerator_kwargs"]["ds_config"]["zero_optimization"]["stage"]
        return level == 3
    except Exception:
        return False
