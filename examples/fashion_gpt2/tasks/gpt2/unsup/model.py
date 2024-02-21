import math
import os
import time
from typing import Dict, List, Optional

import torch
from cruise import CruiseCLI, CruiseConfig, CruiseModule, last_cli
from cruise.utilities.distributed import DIST_ENV
from torch import nn

try:
    from mariana.data.gpt.datamodule.qa_finetune import QAFinetuneGPTDatamodule
    from mariana.data.gpt.datamodule.unsupervised import UnsupGPTDatamodule
    from mariana.models.gpt2 import Conv1D, GPT2LMHeadModel, get_subsequent_mask
    from mariana.optim import mariana_optimizer_kwargs_defaults
    from mariana.utils.checkpoint_utils import is_zero3, load_zero3_state_dict
    from mariana.utils.exp_helper import ExpHelper
    from mariana.utils.generate import play_console, play_file, play_file_qa
except:
    from examples.fashion_gpt2.mariana.data.gpt.datamodule.qa_finetune import QAFinetuneGPTDatamodule
    from examples.fashion_gpt2.mariana.data.gpt.datamodule.unsupervised import UnsupGPTDatamodule
    from examples.fashion_gpt2.mariana.models.gpt2 import Conv1D, GPT2LMHeadModel, get_subsequent_mask
    from examples.fashion_gpt2.mariana.optim import mariana_optimizer_kwargs_defaults
    from examples.fashion_gpt2.mariana.utils.checkpoint_utils import is_zero3, load_zero3_state_dict
    from examples.fashion_gpt2.mariana.utils.exp_helper import ExpHelper
    from examples.fashion_gpt2.mariana.utils.generate import play_console, play_file, play_file_qa

# Config adapter
network_config = {
    "hidden_size": 2048,
    "n_embed": 512,  # vocab embedding
    "n_inner": 8192,
    "n_head": 16,
    "n_layer": 24,
    "vocab_size": 145664,
    "max_position_embeddings": 2048,
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
    "gradient_checkpointing_ln": False,
    "gradient_checkpointing_mlp": False,
    "gradient_checkpointing_start_layers": 0,
    "tie_weight": True,
    "pad_idx": 2,
    "use_ft_flash_attn": False,
    "use_ft_linear": False,
    "use_ft_layernorm": False,
    "use_rmpad": False,
    "pad_output": False,
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
        self.init_weights()
        self.freeze_params(self.hparams.freeze_prefix or [])
        self.average_token_rate = 0
        self.consume_tokens = 0
        self.use_rmpad = network.get("use_rmpad", False)
        self.pad_output = network.get("pad_output", False)
        self.resume_training_step = 0
        self.last_consume_tokens = -1
        self.last_time = time.time()

    def on_train_start(self) -> None:
        global_config = last_cli().hparams
        if global_config["trainer"]["resume_ckpt_path"] and global_config["data"]["dyn_bsz"]:
            self.resume_training_step = self.trainer._current_step
            self.rank_zero_info(f"Resume Training Step: {self.resume_training_step}.")
            coeff = 2
            # Do not support cross-epoch restoration.
            self.trainer._current_step = max(
                0,
                self.trainer._current_step
                - global_config["data"]["train_num_workers"] * global_config["data"]["max_seq_len"] * coeff,
            )
        return super().on_train_start()

    def setup(self):
        # In DDP rank 0 load pretrain weights is enough
        if self.trainer.global_rank == 0 and self.hparams.partial_pretrain:
            rename_params = self.hparams.partial_pretrain_rename or {}
            if "mp_rank" in self.hparams.partial_pretrain:
                # zero2 checkpoints has key 'module'
                from cruise.utilities.cloud_io import load as crs_load

                state_dict = crs_load(self.hparams.partial_pretrain, map_location="cpu")["module"]
                state_dict = {k[7:]: v for k, v in state_dict.items()}
                self.partial_load_from_checkpoints(state_dict, rename_params=rename_params)
            else:
                self.partial_load_from_checkpoints(
                    self.hparams.partial_pretrain,
                    rename_params=rename_params,
                    verbose=True,
                )

    def freeze_params(self, freeze_prefix):
        for name, param in self.named_parameters():
            for prefix in freeze_prefix:
                if name.startswith(prefix):
                    self.rank_zero_print("freeze_params:", name)
                    param.requires_grad = False

    def count_tokens(self, attention_mask):
        batch_tokens = attention_mask.sum().item()
        self.consume_tokens += batch_tokens * self.trainer.world_size
        train_batch_size = self.trainer._datamodule.hparams.train_batch_size
        bsz_warmup_rate = self.trainer._datamodule.hparams.bsz_warmup_rate
        train_steps = self.trainer._datamodule.train_steps
        if self.trainer._datamodule.hparams.dyn_bsz:
            bsz = min(
                (self.trainer.global_step + 1) / bsz_warmup_rate * train_steps,
                train_batch_size,
            )
        else:
            bsz = train_batch_size
        seq_len = self.trainer._datamodule.hparams.max_seq_len

        self.average_token_rate = (
            self.average_token_rate * self.trainer.global_step + batch_tokens / bsz / seq_len
        ) / (self.trainer.global_step + 1)

        if self.trainer.global_step % self.trainer._log_every_n_steps == 0 and self.trainer.global_rank == 0:
            cur_time = time.time()
            tokens_per_second = (self.consume_tokens - self.last_consume_tokens) / (cur_time - self.last_time)
            self.last_time = cur_time
            self.last_consume_tokens = self.consume_tokens

            self.trainer.logger.log_metrics(
                {"consume_tokens(B)": self.consume_tokens / 1e9},
                step=self.trainer.global_step,
                step_size=None,
                reduce_fx=lambda x: x[0] if isinstance(x, list) else x,
                pause_flush=True,
            )
            self.trainer.logger.log_metrics(
                {"average_token_rate": self.average_token_rate * 100},
                step=self.trainer.global_step,
                step_size=None,
                reduce_fx=lambda x: x[0] if isinstance(x, list) else x,
                pause_flush=True,
            )
            self.trainer.logger.log_metrics(
                {"tokens_per_second(M)": tokens_per_second / 1e6},
                step=self.trainer.global_step,
                step_size=None,
                reduce_fx=lambda x: x[0] if isinstance(x, list) else x,
                pause_flush=True,
            )

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
    ):
        self.count_tokens(attention_mask)
        attention_mask = get_subsequent_mask(attention_mask)
        model_out = self.gpt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_rmpad=self.use_rmpad,
            pad_output=self.pad_output,
        )
        if hasattr(self, "_model_stats"):
            self._model_stats["last_activation_norm"] = model_out["last_activation_norm"]
        return model_out

    def on_train_batch_start(self, batch, batch_idx):
        if batch_idx < self.resume_training_step:
            self.trainer._current_step = batch_idx
            if batch_idx % 200 == 0:
                self.rank_zero_info(
                    f"Resuming Data State through iterating data: {batch_idx} / {self.resume_training_step}"
                )
            return -3  # Skip current batch

    def training_step(self, batch, batch_idx):
        # log lr
        scheduler = self.trainer.lr_scheduler_configs[0].scheduler
        if hasattr(scheduler, "get_lr"):
            self.log("lr", scheduler.get_lr()[0], console=True)
        else:
            self.log("lr", scheduler.get_last_lr()[0], console=True)
        # in hf model, labels will be shifted by 1, so here labels = input_ids
        batch["labels"] = batch["input_ids"]
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
            # module.weight.data.normal_(mean=0.0, std=self.hparams.network.initializer_range)
            # 100k vocab, 4096 embed_dim => 0.00438
            module.weight.data.normal_(
                mean=0.0,
                std=math.sqrt(2 / (self.hparams.network.vocab_size + self.hparams.network.n_embed)),
            )
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if "c_proj.weight" in name or "attn_ow" in name:  # deepspeed transformer kernel 是 attn_ow
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(
                    mean=0.0,
                    std=self.hparams.network.initializer_range / math.sqrt(2 * self.hparams.network.n_layer),
                )
            if "pre_token_proj.weight" in name:
                p.data.normal_(
                    mean=0.0,
                    std=math.sqrt(2 / (self.hparams.network.n_embed + self.hparams.network.hidden_size)),
                )

            if "post_token_proj.weight" in name:
                p.data.normal_(
                    mean=0.0,
                    std=math.sqrt(2 / (self.hparams.network.n_embed + self.hparams.network.hidden_size)),
                )


if __name__ == "__main__":
    helper = ExpHelper(__file__)
    from cruise.trainer.callback import ModelCheckpoint

    ckpter = ModelCheckpoint(
        monitor="step",
        save_last=False,
        save_top_k=-1,
        every_n_train_steps=int(os.environ.get("MARIANA_CUSTOM_SAVE_INTERVAL", 10000)),
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        enable_trace=False,
    )

    cli = CruiseCLI(
        GPT2Model,
        datamodule_class=UnsupGPTDatamodule,
        trainer_defaults={
            "precision": 16,
            "enable_versions": False,
            "find_unused_parameters": False,
            "max_epochs": 1,
            "default_hdfs_dir": helper.hdfs_prefix,
            "project_name": helper.project_name,
            "val_check_interval": -1,
            "summarize_model_depth": 2,
            "gradient_clip_val": 1.0,
            "checkpoint_monitor": "step",
            "checkpoint_mode": "max",
            "callbacks": [ckpter],
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
    cli.add_argument("--output_file_path", default=None, help="play file output file path")
    cli.add_argument(
        "--play-file-type",
        default="none",
        type=str,
        help="generate by samples loaded from file for qa",
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
    cli.add_argument("--generate-topk", default=None, type=int, help="sample top-k")
    cli.add_argument(
        "--generate-topp",
        default=None,
        type=float,
        help="sample at least top-p probability",
    )
    cli.add_argument(
        "--generate-dynamic-topp",
        default=None,
        type=float,
        help="sample at least dynamic top-p probability",
    )
    cli.add_argument(
        "--generate-dynamic-topp-omega",
        default=0.3,
        type=float,
        help="omega for dynamic topp",
    )
    cli.add_argument(
        "--generate-dynamic-topp-decay",
        default=0.9,
        type=float,
        help="lambda for dynamic toppp",
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
            if cfg.play_file_type == "qa":
                play_file_qa(
                    cfg.play_file,
                    tokenizer,
                    model.cuda(),
                    output_file_path=cfg.output_file_path,
                    trial_num=cfg.generate_trial_num,
                    steps=cfg.generate_steps,
                    temperature=cfg.generate_temp,
                    do_sample=cfg.generate_do_sample,
                    top_k=cfg.generate_topk,
                    top_p=cfg.generate_topp,
                    dynamic_top_p=cfg.generate_dynamic_topp,
                    omega=cfg.generate_dynamic_topp_omega,
                    decay_lambda=cfg.generate_dynamic_topp_decay,
                    until_n_eos=cfg.generate_n_eos,
                    limit_samples=cfg.play_file_limit,
                )
            else:
                play_file(
                    cfg.play_file,
                    tokenizer,
                    model.cuda(),
                    output_file_path=cfg.output_file_path,
                    trial_num=cfg.generate_trial_num,
                    steps=cfg.generate_steps,
                    temperature=cfg.generate_temp,
                    do_sample=cfg.generate_do_sample,
                    top_k=cfg.generate_topk,
                    top_p=cfg.generate_topp,
                    until_n_eos=cfg.generate_n_eos,
                    dynamic_top_p=cfg.generate_dynamic_topp,
                    omega=cfg.generate_dynamic_topp_omega,
                    decay_lambda=cfg.generate_dynamic_topp_decay,
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
                dynamic_top_p=cfg.generate_dynamic_topp,
                omega=cfg.generate_dynamic_topp_omega,
                decay_lambda=cfg.generate_dynamic_topp_decay,
                until_n_eos=cfg.generate_n_eos,
            )
    else:
        trainer.callbacks = [TokenCheckpointHook()] + trainer.callbacks
        trainer.fit(model, datamodule)
