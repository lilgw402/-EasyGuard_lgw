# -*- coding: utf-8 -*-
import math
from typing import Dict, List, Optional

import torch
from cruise import CruiseCLI, CruiseConfig, CruiseModule, last_cli
from cruise.utilities.distributed import DIST_ENV
from torch import nn

try:
    from mariana.data.gpt.datamodule.qa_finetune import QAFinetuneGPTDatamodule
    from mariana.models.gpt2 import Conv1D, GPT2LMHeadModel, get_subsequent_mask
    from mariana.optim import mariana_optimizer_kwargs_defaults
    from mariana.utils.checkpoint_utils import is_zero3, load_zero3_state_dict
    from mariana.utils.exp_helper import ExpHelper
    from mariana.utils.generate import play_console, play_file
except:  # noqa: E722
    from examples.fashion_gpt2.mariana.data.gpt.datamodule.qa_finetune import QAFinetuneGPTDatamodule
    from examples.fashion_gpt2.mariana.models.gpt2 import Conv1D, GPT2LMHeadModel, get_subsequent_mask
    from examples.fashion_gpt2.mariana.optim import mariana_optimizer_kwargs_defaults
    from examples.fashion_gpt2.mariana.utils.checkpoint_utils import is_zero3, load_zero3_state_dict
    from examples.fashion_gpt2.mariana.utils.exp_helper import ExpHelper
    from examples.fashion_gpt2.mariana.utils.generate import play_console, play_file

# Config adapter
network_config = {
    "hidden_size": 2048,
    "n_embed": 512,  # vocab embedding
    "n_inner": 8192,
    "n_head": 16,
    "n_layer": 24,
    "vocab_size": 145664,
    "max_position_embeddings": 1025,
    "layer_norm_epsilon": 1.0e-5,
    "activation_function": "gelu_new",
    "resid_pdrop": 0.1,
    "embd_pdrop": 0.1,
    "attn_pdrop": 0.1,
    "scale_attn_weights": True,  # TODO:
    "scale_attn_by_inverse_layer_idx": False,  # TODO:
    "reorder_and_upcast_attn": False,  # TODO:
    "initializer_range": 0.02,
    "gradient_checkpointing": False,
    "tie_weight": True,
    "pad_idx": 2,
    "use_ft_flash_attn": False,
    "use_ft_linear": False,
    "use_ft_layernorm": False,
    "use_rmpad": False,
}


class GPT2Model(CruiseModule):
    """Deberta pretrain"""

    def __init__(
        self,
        network: CruiseConfig = network_config,
        freeze_prefix: Optional[List[str]] = None,
        partial_pretrain: Optional[str] = None,
        partial_pretrain_rename: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self.save_hparams()  # save to self.hparams

        # 文本
        self.gpt = GPT2LMHeadModel(self.hparams)
        # self.init_weights()  # will always load pretrained weights
        self.freeze_params(self.hparams.freeze_prefix or [])

    def setup(self):
        _is_zero3 = is_zero3(last_cli().hparams)
        # In DDP rank 0 load pretrain weights is enough
        if (_is_zero3 or self.trainer.global_rank == 0) and self.hparams.partial_pretrain:
            rename_params = self.hparams.partial_pretrain_rename or {}
            from cruise.utilities.cloud_io import load as crs_load

            state_dict = crs_load(self.hparams.partial_pretrain, map_location="cpu")
            if "mp_rank" in self.hparams.partial_pretrain:
                # zero2 checkpoints has key 'module'
                state_dict = state_dict["module"]
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            if _is_zero3:
                metadata = getattr(state_dict, "_metadata", None)
                error_msgs = []
                load_zero3_state_dict(state_dict, self, metadata, error_msgs, prefix="")
            else:
                self.partial_load_from_checkpoints(state_dict, rename_params=rename_params, verbose=True)

    def freeze_params(self, freeze_prefix):
        for name, param in self.named_parameters():
            for prefix in freeze_prefix:
                if name.startswith(prefix):
                    self.rank_zero_print("freeze_params:", name)
                    param.requires_grad = False

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
    ):
        attention_mask = get_subsequent_mask(attention_mask)
        model_out = self.gpt(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return model_out

    def training_step(self, batch, batch_idx):
        # log lr
        scheduler = self.trainer.lr_scheduler_configs[0].scheduler
        if hasattr(scheduler, "get_lr"):
            self.log("lr", scheduler.get_lr()[0], console=True)
        else:
            self.log("lr", scheduler.get_last_lr()[0], console=True)
        # in hf model, labels will be shifted by 1, so here labels = input_ids
        batch["labels"] = batch["input_ids"]
        if "loss_mask" in batch:
            # we mask loss to non-prompt part by setting prompt label to pad_idx
            batch["labels"] = batch["labels"].clone().detach()
            loss_mask = batch.pop("loss_mask")
            batch["labels"][torch.where(loss_mask == 0)] = self.hparams.network.pad_idx
            del loss_mask
            loss_mask = None
        model_out = self.forward(**batch)
        loss = model_out["loss"]
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # in hf model, labels will be shifted by 1, so here labels = input_ids
        batch["labels"] = batch["input_ids"]
        model_out = self.forward(**batch)
        loss = model_out["loss"]
        return {"val_loss": loss}

    @torch.no_grad()
    def decode(
        self,
        input_ids: torch.Tensor,
        input_mask: torch.Tensor,
        *args,
        **kwargs,
    ):
        """For generation task"""
        model_out = self.gpt(input_ids=input_ids, attention_mask=input_mask)
        return model_out

    def configure_optimizers(self, optimizer_kwargs):
        """
        Model定制optimizer和lr_scheduler
        """
        no_decay = ["bias", "bn", "norm", "ln"]
        no_dacay_params_dict = {"params": [], "weight_decay": 0.0}
        normal_params_dict = {
            "params": [],
            "weight_decay": optimizer_kwargs["optimizer"]["params"]["weight_decay"],
        }

        for n, p in self.named_parameters():
            if any(nd in n for nd in no_decay):
                no_dacay_params_dict["params"].append(p)
            else:
                normal_params_dict["params"].append(p)
        optimizer_grouped_parameters = [
            no_dacay_params_dict,
            normal_params_dict,
        ]

        optimizers = super()._configure_optimizers(optimizer_grouped_parameters, optimizer_kwargs)
        lr_schedulers = super()._configure_schedulers(optimizers, optimizer_kwargs)
        return optimizers, lr_schedulers

    def lr_scheduler_step(
        self,
        schedulers,
        **kwargs,
    ) -> None:
        r"""
        默认是per epoch的lr schedule, 改成per step的
        """
        # if self.trainer.global_step == 0:
        #     # skip first step
        #     return
        for scheduler in schedulers:
            scheduler.step()

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.hparams.network.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.hparams.network.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale    # noqa: E501
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if "c_proj.weight" in name or "attn_ow" in name:  # deepspeed transformer kernel 是 attn_ow
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(
                    mean=0.0,
                    std=(self.hparams.network.initializer_range / math.sqrt(2 * self.hparams.network.n_layer)),
                )


if __name__ == "__main__":
    helper = ExpHelper(__file__)
    from cruise.trainer.callback import ModelCheckpoint

    checkpointer = ModelCheckpoint(
        monitor="epoch",
        save_last=True,
        save_top_k=-1,
        every_n_train_steps=2000,
        save_weights_only=True,
        save_on_train_epoch_end=True,
        enable_trace=False,
    )
    cli = CruiseCLI(
        GPT2Model,
        datamodule_class=QAFinetuneGPTDatamodule,
        trainer_defaults={
            "precision": 16,
            "enable_versions": False,
            "log_every_n_steps": 100,
            "find_unused_parameters": False,
            "max_epochs": 10,
            "resume_ckpt_path": None,
            "default_hdfs_dir": helper.hdfs_prefix,
            "project_name": helper.project_name,
            "val_check_interval": -1,
            "summarize_model_depth": 2,
            "gradient_clip_val": 1.0,
            "checkpoint_monitor": "step",
            "checkpoint_mode": "max",
            "callbacks": [checkpointer],
            "optimizer_kwargs": mariana_optimizer_kwargs_defaults,
        },
    )
    cli.add_argument("--val-only", default=False, action="store_true", dest="val_only")
    cli.add_argument("--play", default=False, action="store_true", dest="play")
    cli.add_argument(
        "--play-file",
        default="",
        type=str,
        help="generate by samples loaded from file",
    )
    cli.add_argument(
        "--play-file-limit",
        default=-1,
        type=int,
        help="If >0, limit how many lines to generate.",
    )
    cli.add_argument(
        "--generate-trial-num",
        default=5,
        type=int,
        help="generation trial num, default is 5",
    )
    cli.add_argument(
        "--generate-steps",
        default=256,
        type=int,
        help="decode sequence length/steps",
    )
    cli.add_argument(
        "--generate-temp",
        default=0.7,
        type=float,
        help="Smaller tempreature logits become more steep",
    )
    cli.add_argument(
        "--generate-do-sample",
        default=True,
        type=bool,
        help="multinomial sample if True",
    )
    cli.add_argument("--generate-topk", default=5, type=int, help="sample top-k")
    cli.add_argument(
        "--generate-topp",
        default=None,
        type=float,
        help="sample at least top-p probability",
    )
    cli.add_argument("--generate-n-eos", default=1, type=int, help="Stop until n-eos tokens")

    cfg, trainer, model, datamodule = cli.parse_args()
    if cfg.val_only:
        trainer.validate(model, datamodule=datamodule)
    elif cfg.play_file or cfg.play:
        assert DIST_ENV.world_size == 1, "Play mode only support single card"
        datamodule.rank_zero_prepare()
        datamodule.local_rank_zero_prepare()
        datamodule.setup()
        tokenizer = datamodule.tokenizer
        assert tokenizer is not None, "Invalid tokenizer from datamodule"
        model.rank_zero_prepare()
        model.local_rank_zero_prepare()
        model.setup()
        if cfg.play_file:
            print("\nFile play mode.")
            play_file(
                cfg.play_file,
                tokenizer,
                model.cuda(),
                cfg.generate_trial_num,
                steps=cfg.generate_steps,
                temperature=cfg.generate_temp,
                do_sample=cfg.generate_do_sample,
                top_k=cfg.generate_topk,
                top_p=cfg.generate_topp,
                until_n_eos=cfg.generate_n_eos,
                limit_samples=cfg.play_file_limit,
            )
        else:
            print("\nConsole play mode.")
            play_console(
                tokenizer,
                model.cuda(),
                cfg.generate_trial_num,
                steps=cfg.generate_steps,
                temperature=cfg.generate_temp,
                do_sample=cfg.generate_do_sample,
                top_k=cfg.generate_topk,
                top_p=cfg.generate_topp,
                until_n_eos=cfg.generate_n_eos,
            )
    else:
        trainer.fit(model, datamodule)
