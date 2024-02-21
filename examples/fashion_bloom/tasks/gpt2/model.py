import math
import os
import tempfile
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from cruise import CruiseCLI, CruiseConfig, CruiseModule
from cruise.utilities.distributed import DIST_ENV
from cruise.utilities.hdfs_io import hcopy
from fashBloom.data.gpt.datamodule.zero_shot import ZeroShotGPTDatamodule
from fashBloom.optim import mariana_optimizer_kwargs_defaults
from fashBloom.utils.exp_helper import ExpHelper
from fashBloom.utils.generate import (
    few_shot_play_file,
    play_console,
    play_file,
    play_file_qa,
    play_file_qa_batch,
    play_file_qa_batch_from_offical_generate,
)
from sklearn.covariance import log_likelihood
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

# from transformers import LlamaTokenizer
from transformers.deepspeed import HfDeepSpeedConfig


def get_subsequent_mask(seq):
    """
    For masking out the subsequent info.
    seq: [bsz, seq_len]
    mask: [bsz, seq_len, seq_len]
    """
    _, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    mask = seq.unsqueeze(-2) & subsequent_mask
    return mask


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
    Basically works like a linear layer but the weights are transposed.
    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


# Config adapter
network_config = {
    "hidden_size": 2048,
    "n_embed": 2048,  # 512,  # vocab embedding
    "n_inner": 8192,
    "n_head": 16,
    "n_layer": 24,
    "vocab_size": 145664,
    "max_position_embeddings": 2048,  # 1025,
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
    "pad_idx": 3,
    "use_ft_flash_attn": False,
    "use_ft_linear": False,
    "use_ft_layernorm": False,
    "use_rmpad_lmloss": False,
    "use_rmpad_lnmlp": False,
    "use_rmpad_attn": False,
}


class GPT2Model(CruiseModule):
    """Deberta pretrain"""

    def __init__(
        self,
        network: CruiseConfig = network_config,
        freeze_prefix: Optional[List[str]] = None,
        partial_pretrain: Optional[str] = None,
        partial_pretrain_rename: Optional[Dict[str, str]] = None,
        use_hf_ckpt: Optional[bool] = False,
        model_config: Optional[str] = None,
    ):
        super().__init__()
        self.save_hparams()  # save to self.hparams
        # TODO: check if this is needed
        self.best_val_performance_value = 0.0

        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.hparams.network.pad_idx)

        if self.hparams.use_hf_ckpt:
            if self.hparams.partial_pretrain.startswith("hdfs"):
                tmp_dir = os.path.join(
                    tempfile.gettempdir(),
                    os.path.basename(self.hparams.partial_pretrain),
                )
                self.local_dir = tmp_dir
            else:
                self.local_dir = self.hparams.partial_pretrain
            print(f"local_dir: {self.local_dir}")

            if self.hparams.model_config.startswith("hdfs"):
                self.local_config = self.local_dir
            else:
                self.local_config = self.hparams.model_config
            print(f"local_config: {self.local_config}")
        else:
            if self.hparams.partial_pretrain.startswith("hdfs"):
                tmp_dir = os.path.join(
                    tempfile.gettempdir(),
                    os.path.basename(self.hparams.partial_pretrain),
                )
                self.local_dir = tmp_dir
            else:
                self.local_dir = self.hparams.partial_pretrain
            print(f"local_dir: {self.local_dir}")

            if self.hparams.model_config.startswith("hdfs"):
                self.local_config = os.path.join(
                    tempfile.gettempdir(),
                    os.path.basename(self.hparams.model_config),
                )
            else:
                self.local_config = self.hparams.model_config
            print(f"local_config: {self.local_config}")

    def local_rank_zero_prepare(self) -> None:
        if self.hparams.use_hf_ckpt:
            if self.hparams.partial_pretrain.startswith("hdfs"):
                hcopy(self.hparams.partial_pretrain, self.local_dir)
            if self.hparams.model_config.startswith("hdfs") and self.local_dir != self.local_config:
                hcopy(self.hparams.model_config, self.local_config)
        else:
            if self.hparams.partial_pretrain.startswith("hdfs"):
                hcopy(self.hparams.partial_pretrain, self.local_dir)
            if self.hparams.model_config.startswith("hdfs") and self.local_dir != self.local_config:
                hcopy(self.hparams.model_config, self.local_config)

    def setup(self):
        if self.hparams.use_hf_ckpt:
            self.hf_config = AutoConfig.from_pretrained(self.local_config)
            print(f"self.local_config: {self.local_config}")
            print(f"self.hf_config: {self.hf_config}")
            self.hf_config.gradient_checkpointing = True
            self.hf_config.use_cache = False

            if not self.hparams.partial_pretrain:
                self.gpt = AutoModelForCausalLM.from_config(config=self.hf_config)
                self.freeze_params(self.hparams.freeze_prefix or [])
        else:
            self.hf_config = AutoConfig.from_pretrained(self.local_config)
            print(f"self.local_config: {self.local_config}")
            print(f"self.hf_config: {self.hf_config}")
            self.hf_config.gradient_checkpointing = True
            self.hf_config.use_cache = False

            self.gpt = AutoModelForCausalLM.from_config(config=self.hf_config)
            self.gpt.lm_head.weight = self.gpt.transformer.word_embeddings.weight
            # self._tie_or_clone_weights(self.gpt.lm_head.weight, self.gpt.transformer.word_embeddings.weight)
            # self.gpt.lm_head.weight = nn.Parameter(self.gpt.transformer.word_embeddings.weight.clone())

            self.freeze_params(self.hparams.freeze_prefix or [])

        self.rank_zero_print("xxxxx starting loading checkpoint")
        # In DDP rank 0 load pretrain weights is enough
        # if self.trainer.global_rank == 0 and self.hparams.partial_pretrain:
        if self.hparams.partial_pretrain:
            rename_params = self.hparams.partial_pretrain_rename or {}
            if self.hparams.use_hf_ckpt:
                self.rank_zero_print(f"load from load_from_hf")
                self.gpt = AutoModelForCausalLM.from_pretrained(self.local_dir, config=self.hf_config)

                self.rank_zero_print("MODEL ARCH: ")
                self.rank_zero_print(self.gpt)

                self.freeze_params(self.hparams.freeze_prefix or [])
            elif "mp_rank" in self.hparams.partial_pretrain:
                self.rank_zero_print("******** zero2... ********")
                # zero2 checkpoints has key 'module'
                from cruise.utilities.cloud_io import load as crs_load

                state_dict = crs_load(self.hparams.partial_pretrain, map_location="cpu")["module"]
                state_dict = {k[7:]: v for k, v in state_dict.items()}
                self.partial_load_from_checkpoints(state_dict, rename_params=rename_params)
                self.gpt.lm_head.weight = self.gpt.transformer.word_embeddings.weight
            else:
                # zero3
                self.rank_zero_print("******** zero3... ********")

                self.partial_load_from_checkpoints(
                    self.hparams.partial_pretrain,
                    rename_params=rename_params,
                    verbose=True,
                )

                self.gpt.lm_head.weight = self.gpt.transformer.word_embeddings.weight
                # self.gpt.lm_head.weight = nn.Parameter(self.gpt.transformer.word_embeddings.weight.clone())

        self.rank_zero_print("xxxxx finished loading checkpoint")

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
        loss_mask=None,
        dataset_name=None,
        task_name=None,
        input_lens=None,
        target_idx=None,
        answer_choice_tokens_list=None,
    ):
        if not self.hparams.use_hf_ckpt and labels is None:
            # print(f'before_attention_mask: {attention_mask.size()}')
            attention_mask = get_subsequent_mask(attention_mask)
            # print(f'after_attention_mask: {attention_mask.size()}')

        # TODO: log, remove later
        # print("print: attention_mask={}, \n input_ids={}, \n labels={} \n".format(attention_mask, input_ids, labels))
        # print(attention_mask.shape, input_ids.shape, labels.shape)
        # print(attention_mask[0, :100])
        # print(input_ids[0, :100])

        model_out = self.gpt(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # print("output=", model_out)
        # print(model_out['logits'].shape)

        return model_out

    def training_step(self, batch, batch_idx):
        # log lr
        # (TODO) Need to confirm if we still need lr scheduler
        scheduler = self.trainer.lr_scheduler_configs[0].scheduler
        if hasattr(scheduler, "get_lr"):
            self.log("lr", scheduler.get_lr()[0], console=True)
        else:
            self.log("lr", scheduler.get_last_lr()[0], console=True)

        # in hf model, labels will be shifted by 1, so here labels = input_ids
        batch["labels"] = batch["input_ids"]
        # TODO: remove debug log below
        # print("batch information: {}, with batch_idx: {}".format(batch, batch_idx))
        # self.rank_zero_print("batch information: {}, with batch_idx: {}".format(batch, batch_idx))

        model_out = self.forward(**batch)
        loss = None

        if "loss_mask" in batch:
            batch["labels"] = batch["labels"].clone().detach()
            loss_mask = batch.pop("loss_mask")
            batch["labels"][torch.where(loss_mask == 0)] = self.hparams.network.pad_idx
            del loss_mask
            loss_mask = None

            # Shift so that tokens < n predict n
            shift_logits = model_out["logits"][..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss = self.loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size),
                shift_labels.view(batch_size * seq_length),
            )
            # print('my loss')
            # print(loss)
        else:
            loss = model_out["loss"]

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # in hf model, labels will be shifted by 1, so here labels = input_ids
        batch["labels"] = batch["input_ids"]
        model_out = self.forward(**batch)
        loss = model_out["loss"]
        return {"val_loss": loss}

    # def validation_epoch_end(self, outputs):
    #     rank_total_sample = torch.as_tensor([out['num_sample'] for out in outputs])
    #     all_rank_total_sample_list = self.all_gather(rank_total_sample, sync_grads=False)
    #     all_rank_total_sample = [sum(num) for num in all_rank_total_sample_list]

    #     rank_total_correct = torch.as_tensor([out['val_loss'] for out in outputs])

    #     all_rank_total_correct_list = self.all_gather(rank_total_correct, sync_grads=False)
    #     all_rank_total_correct = [sum(num) for num in all_rank_total_correct_list]

    #     acc = sum(all_rank_total_correct) * 1.0 / sum(all_rank_total_sample)

    #     if acc >= self.best_val_performance_value:
    #         self.best_val_performance_value = acc

    #     self.rank_zero_info(f"Zero Shot Learning Acc: all_rank_total_sample:{sum(all_rank_total_sample)}, all_rank_total_loss:{sum(all_rank_total_correct)}, avg_loss of current step: {acc}")
    #     self.rank_zero_info(f"Zero Shot Learning Acc: best val_acc_per_epoch so far: {self.best_val_performance_value}")

    #     self.log_dict({
    #         f'avg_loss': acc,
    #         f'num_sample': sum(all_rank_total_sample),
    #         f'all_loss': sum(all_rank_total_correct)
    #     }, console=True)

    #     return acc

    @torch.no_grad()
    def decode(
        self,
        input_ids: torch.Tensor,
        input_mask: torch.Tensor,
        *args,
        **kwargs,
    ):
        """For generation task"""
        # TODO: remove log later
        # print("print: decode being called")

        model_out = self.gpt(input_ids=input_ids, attention_mask=input_mask)
        return model_out

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
                    std=(self.hparams.network.initializer_range / math.sqrt(2 * self.hparams.network.n_layer)),
                )


if __name__ == "__main__":
    helper = ExpHelper(__file__)
    from cruise.trainer.callback import ModelCheckpoint

    ckpter = ModelCheckpoint(
        monitor="step",
        save_last=False,
        save_top_k=-1,
        every_n_train_steps=5000,
        every_n_epochs=1,
        save_on_train_epoch_end=False,
        enable_trace=False,
    )
    cli = CruiseCLI(
        GPT2Model,
        datamodule_class=ZeroShotGPTDatamodule,
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
    cli.add_argument(
        "--play-out-file",
        default="",
        type=str,
        help="generate by samples loaded from file",
        dest="play_out_file",
    )
    cli.add_argument(
        "--dataset-name",
        default="",
        type=str,
        help="dataset name",
        dest="dataset_name",
    )
    cli.add_argument(
        "--subset-name",
        default="",
        type=str,
        help="subset name",
        dest="subset_name",
    )
    cli.add_argument(
        "--template-name",
        default="",
        type=str,
        help="template name",
        dest="template_name",
    )
    cli.add_argument(
        "--play-file-type",
        default="none",
        type=str,
        help="generate by samples loaded from file for qa",
    )
    cli.add_argument(
        "--play-file-bsz",
        default=4,
        type=int,
        help="val batch size for inference",
    )
    cli.add_argument(
        "--play-file-limit",
        default=-1,
        type=int,
        help="If >0, limit how many lines to generate.",
    )
    cli.add_argument(
        "--generate-trial-num",
        default=1,
        type=int,
        help="generation trial num, default is 5",
    )
    cli.add_argument(
        "--generate-steps",
        default=100,
        type=int,
        help="decode sequence length/steps",
    )
    cli.add_argument(
        "--generate-temp",
        default=0.5,
        type=float,
        help="Smaller tempreature logits become more steep",
    )
    cli.add_argument(
        "--generate-do-sample",
        default=False,
        type=bool,
        help="multinomial sample if True",
    )
    cli.add_argument("--generate-topk", default=1, type=int, help="sample top-k")
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
    cli.add_argument("--num-fewshot", default=0, type=int, help="fewshot to control")
    cli.add_argument("--fewshot-file-path", default="", type=str, help="few shot file path")
    cli.add_argument(
        "--generate-policy",
        default=False,
        action="store_true",
        help="decode from .generate or custom policy",
    )

    cfg, trainer, model, datamodule = cli.parse_args()

    try:
        hfds_config_file_or_dict = cfg["trainer"]["accelerator_kwargs"]["ds_config"]
        hfds_config = HfDeepSpeedConfig(hfds_config_file_or_dict)
    except:
        pass

    if cfg.val_only:
        trainer.validate(model, datamodule=datamodule)
    elif cfg.play_file or cfg.play:
        # assert DIST_ENV.world_size == 1, "Play mode only support single card"
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
            if cfg.play_file_type == "qa_batch":
                if not cfg.generate_policy:
                    print("\nqa_batch: custom")
                    play_file_qa_batch(
                        cfg.play_file,
                        cfg.play_out_file,
                        tokenizer,
                        model.cuda(),
                        cfg.generate_trial_num,
                        val_bsz=cfg.play_file_bsz,
                        steps=cfg.generate_steps,
                        temperature=cfg.generate_temp,
                        do_sample=cfg.generate_do_sample,
                        top_k=cfg.generate_topk,
                        top_p=cfg.generate_topp,
                        until_n_eos=cfg.generate_n_eos,
                        limit_samples=cfg.play_file_limit,
                    )
                else:
                    print("\nqa_batch: generate by generate_policy")
                    play_file_qa_batch_from_offical_generate(
                        cfg.play_file,
                        cfg.play_out_file,
                        tokenizer,
                        model.cuda(),
                        cfg.generate_trial_num,
                        val_bsz=cfg.play_file_bsz,
                        steps=cfg.generate_steps,
                        temperature=cfg.generate_temp,
                        do_sample=cfg.generate_do_sample,
                        top_k=cfg.generate_topk,
                        top_p=cfg.generate_topp,
                        until_n_eos=cfg.generate_n_eos,
                        limit_samples=cfg.play_file_limit,
                    )
            elif cfg.play_file_type == "qa":
                play_file_qa(
                    cfg.play_file,
                    cfg.play_out_file,
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
                few_shot_play_file(
                    cfg.play_file,
                    cfg.play_out_file,
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
                    dataset_name=cfg.dataset_name,
                    subset_name=cfg.subset_name,
                    template_name=cfg.template_name,
                    num_fewshot=cfg.num_fewshot,
                    fewshot_file_path=cfg.fewshot_file_path,
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
