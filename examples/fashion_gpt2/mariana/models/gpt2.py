# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PyTorch OpenAI GPT-2 model.
originally from: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
Copied from fuxi with its config space for compatibility: https://code.byted.org/nlp/fuxi/blob/master/tasks/alice/model/modeling_gpt2.py
"""

import logging
import os
from typing import List, Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from mariana.nn.activations import ACT2FN
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import CrossEntropyLoss

from .xperf_training import FTFlashAttention, FTLayerNorm, FTLinear, FTLinearWeightTransposed, pad_input, unpad_input

# from ...utils.model_parallel_utils import assert_device_map, get_device_map


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


def prune_conv1d_layer(layer: Conv1D, index: torch.LongTensor, dim: int = 1) -> Conv1D:
    """
    Prune a Conv1D layer to keep only entries in index. A Conv1D work as a Linear layer (see e.g. BERT) but the weights
    are transposed.
    Used to remove heads.
    Args:
        layer ([`~pytorch_utils.Conv1D`]): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 1): The dimension on which to keep the indices.
    Returns:
        [`~pytorch_utils.Conv1D`]: The pruned layer as a new layer with `requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if dim == 0:
        b = layer.bias.clone().detach()
    else:
        b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = Conv1D(new_size[1], new_size[0]).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    new_layer.bias.requires_grad = False
    new_layer.bias.copy_(b.contiguous())
    new_layer.bias.requires_grad = True
    return new_layer


def find_pruneable_heads_and_indices(
    heads: List[int],
    n_heads: int,
    head_size: int,
    already_pruned_heads: Set[int],
) -> Tuple[Set[int], torch.LongTensor]:
    """
    Finds the heads and their indices taking `already_pruned_heads` into account.
    Args:
        heads (`List[int]`): List of the indices of heads to prune.
        n_heads (`int`): The number of heads in the model.
        head_size (`int`): The size of each head.
        already_pruned_heads (`Set[int]`): A set of already pruned heads.
    Returns:
        `Tuple[Set[int], torch.LongTensor]`: A tuple with the remaining heads and their corresponding indices.
    """
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    return heads, index


class GPT2Attention(nn.Module):
    # 参考huggingface gpt2代码
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        self.use_ft_flash_attn = config.use_ft_flash_attn
        if self.use_ft_flash_attn:
            assert FTFlashAttention is not None
            self.ft_flash_attn = FTFlashAttention()

        self.attn_pdrop = config.attn_pdrop

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [],
                value.size(-1) ** 0.5,
                dtype=attn_weights.dtype,
                device=attn_weights.device,
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].to(torch.bool)
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        # Preallocate attn_weights for `baddbmm`
        attn_weights = torch.empty(
            bsz * num_heads,
            q_seq_len,
            k_seq_len,
            dtype=torch.float32,
            device=query.device,
        )

        # Compute Scale Factor
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        with autocast(enabled=False):
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        seq_lens: Optional[torch.IntTensor] = None,
        word_idx: Optional[torch.IntTensor] = None,
        use_rmpad: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=-1)
        if not use_rmpad:
            query = self._split_heads(query, self.num_heads, self.head_dim)
            key = self._split_heads(key, self.num_heads, self.head_dim)
            value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            # attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
            if self.use_ft_flash_attn:
                assert self.scale_attn_weights == True and (self.scale_attn_by_inverse_layer_idx in [None, False])
                assert head_mask == None and (output_attentions in [None, False])

                if not use_rmpad:
                    # (batch, head, seq_length, head_features) -> (batch, seq_length, head, head_features)
                    query = query.transpose(-2, -3)
                    key = key.transpose(-2, -3)
                    value = value.transpose(-2, -3)

                    # (batch, seq_length, head, head_features) -> (batch, seq_length, head * head_features)
                    query = query.view(query.shape[0], query.shape[1], -1)
                    key = key.view(key.shape[0], key.shape[1], -1)
                    value = value.view(value.shape[0], value.shape[1], -1)

                # attn_output = self.ft_flash_attn([query, key, value], self.num_heads, attn_mask=attention_mask, causal=True, attention_dropout=self.attn_pdrop)
                attn_output = self.ft_flash_attn(
                    [query, key, value],
                    self.num_heads,
                    causal=True,
                    attention_dropout=self.attn_pdrop,
                    word_idx=word_idx,
                    cu_seqlens_k=seq_lens,
                    cu_seqlens_q=seq_lens,
                    use_rmpad_attn=use_rmpad,
                )
                if not use_rmpad:
                    # (batch, seq_length, head * head_features) -> (batch, seq_length, head, head_features)
                    attn_output = attn_output.view(
                        attn_output.shape[0],
                        attn_output.shape[1],
                        self.num_heads,
                        -1,
                    )
                    # (batch, seq_length, head, head_features) -> (batch, head, seq_length, head_features)
                    attn_output = attn_output.transpose(-2, -3)
            else:
                attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        if not use_rmpad:
            # (batch, head, seq_length, head_features) -> (batch, seq_length, head*head_features)
            attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)  # attn_output
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config, layer_idx=None):
        super().__init__()
        embed_dim = config.hidden_size
        if config.get("use_ft_linear", False) and config.activation_function == "gelu_new":
            assert FTLinearWeightTransposed is not None
            # FTLinearWeightTransposed is using FTLinear internally, but weights are transposed
            self.c_fc = FTLinearWeightTransposed(embed_dim, intermediate_size, act_gelu=True)
            self.c_proj = FTLinearWeightTransposed(intermediate_size, embed_dim)
            self.act = nn.Identity()
        else:
            self.c_fc = Conv1D(intermediate_size, embed_dim)
            self.c_proj = Conv1D(embed_dim, intermediate_size)
            self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)
        self.layer_idx = layer_idx
        self.config = config

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        with torch.no_grad():
            if bool(self.layer_idx) and self.layer_idx == self.config.n_layer - 1:
                last_activation_norm = hidden_states.double().norm(2)
            else:
                last_activation_norm = None
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, last_activation_norm


class GPT2Block(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.gradient_checkpointing_ln = config.get("gradient_checkpointing_ln", False)
        self.gradient_checkpointing_mlp = config.get("gradient_checkpointing_mlp", False)
        gradient_checkpointing_start_layers = config.get("gradient_checkpointing_start_layers", 0)
        if layer_idx < gradient_checkpointing_start_layers:
            self.gradient_checkpointing_ln = False
            self.gradient_checkpointing_mlp = False

        if config.use_ft_layernorm:
            assert FTLayerNorm is not None
            self.ln_1 = FTLayerNorm(hidden_size, eps=config.layer_norm_epsilon)
            self.ln_2 = FTLayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        else:
            self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
            self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config, layer_idx=layer_idx)

        # if config.add_cross_attention:
        #     self.crossattention = GPT2Attention(config, is_cross_attention=True, layer_idx=layer_idx)
        #     self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config, layer_idx)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        word_idx: Optional[torch.IntTensor] = None,
        seq_lens: Optional[torch.IntTensor] = None,
        use_rmpad: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]],]:
        residual = hidden_states
        # https://github.com/cybertronai/gradient-checkpointing
        # https://pytorch.org/docs/stable/checkpoint.html
        if self.gradient_checkpointing_ln:
            hidden_states = torch.utils.checkpoint.checkpoint(self.ln_1, hidden_states)
        else:
            hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            seq_lens=seq_lens,
            word_idx=word_idx,
            use_rmpad=use_rmpad,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights
        residual = hidden_states
        if self.gradient_checkpointing_ln:
            hidden_states = torch.utils.checkpoint.checkpoint(
                self.ln_2,
                hidden_states,
            )
        else:
            hidden_states = self.ln_2(hidden_states)

        if self.gradient_checkpointing_mlp:
            (
                feed_forward_hidden_states,
                last_activation_norm,
            ) = torch.utils.checkpoint.checkpoint(
                self.mlp,
                hidden_states,
            )
        else:
            feed_forward_hidden_states, last_activation_norm = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs + (last_activation_norm,)
        else:
            outputs = (hidden_states,) + outputs[1:] + (last_activation_norm,)

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class GPT2Model(nn.Module):
    "The bare GPT2 Model transformer outputting raw hidden-states without any specific head on top."

    def __init__(self, config):
        super().__init__()
        self.config = config.network
        self.embed_dim = self.config.n_embed

        self.wte = nn.Embedding(self.config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(self.config.max_position_embeddings, self.embed_dim)
        if self.embed_dim != self.config.hidden_size:  # factorization
            if self.config.use_ft_linear:
                assert FTLinear is not None
                self.pre_token_proj = FTLinear(self.embed_dim, self.config.hidden_size)
                self.post_token_proj = FTLinear(self.config.hidden_size, self.embed_dim)
            else:
                self.pre_token_proj = nn.Linear(self.embed_dim, self.config.hidden_size)
                self.post_token_proj = nn.Linear(self.config.hidden_size, self.embed_dim)
        else:
            self.pre_token_proj = None
            self.post_token_proj = None

        self.drop = nn.Dropout(self.config.embd_pdrop)

        self.transformer_kernel = self.config.get("transformer_kernel", "default")
        if self.transformer_kernel == "default":
            self.h = nn.ModuleList([GPT2Block(self.config, layer_idx=i) for i in range(self.config.n_layer)])
        elif self.transformer_kernel == "deepspeed":
            from deepspeed import DeepSpeedTransformerConfig, DeepSpeedTransformerLayer

            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            seed = 42
            ke_config = DeepSpeedTransformerConfig(
                batch_size=config.TRAINER.TRAIN_BATCH_SIZE,  # TODO: hack
                hidden_size=self.config.hidden_size,
                intermediate_size=self.config.n_inner,
                heads=self.config.n_head,
                attn_dropout_ratio=self.config.attn_pdrop,
                hidden_dropout_ratio=self.config.resid_pdrop,
                num_hidden_layers=self.config.n_layer,
                initializer_range=self.config.initializer_range,
                local_rank=local_rank,
                seed=seed,
                fp16=True,
                pre_layer_norm=False,
                attn_dropout_checkpoint=False,
                normalize_invertible=False,
                gelu_checkpoint=False,
                stochastic_mode=False,
            )
            self.h = nn.ModuleList([DeepSpeedTransformerLayer(ke_config) for i in range(self.config.n_layer)])
        elif self.transformer_kernel == "lightseq":
            from lightseq.training.ops.pytorch.gpt_layer import LSHFGptEncoderLayer

            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            ke_config = LSHFGptEncoderLayer.get_config(
                max_batch_tokens=self.config.max_position_embeddings * config.TRAINER.TRAIN_BATCH_SIZE,
                max_seq_len=self.config.max_position_embeddings,
                hidden_size=self.config.hidden_size,
                intermediate_size=self.config.n_inner,
                nhead=self.config.n_head,
                attn_prob_dropout_ratio=self.config.attn_pdrop,
                activation_dropout_ratio=self.config.resid_pdrop,
                hidden_dropout_ratio=self.config.resid_pdrop,
                pre_layer_norm=True,
                activation_fn="gelu",
                fp16=True,
                local_rank=local_rank,
            )
            self.h = nn.ModuleList([LSHFGptEncoderLayer(ke_config) for i in range(self.config.n_layer)])

        if self.config.use_ft_layernorm:
            assert FTLayerNorm is not None
            self.ln_f = FTLayerNorm(self.embed_dim, eps=self.config.layer_norm_epsilon)
        else:
            self.ln_f = nn.LayerNorm(self.embed_dim, eps=self.config.layer_norm_epsilon)

        # Model parallel #TODO: 似乎支持model parallel，先抄过来
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = self.config.get("gradient_checkpointing", False)
        # self.dtype = torch.float16 # TODO:

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        word_idx: Optional[torch.IntTensor] = None,
        use_rmpad: Optional[bool] = False,
    ):
        """ """

        # 1 处理input
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
        device = input_ids.device
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        # past_key_values 可以传
        past_length = 0
        past_key_values = tuple([None] * len(self.h))
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                input_shape[-1] + past_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # move here

        if self.gradient_checkpointing and self.training:
            inputs_embeds = torch.utils.checkpoint.checkpoint(self.wte, input_ids)
            position_embeds = torch.utils.checkpoint.checkpoint(self.wpe, position_ids)
        else:
            inputs_embeds = self.wte(input_ids)
            position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        # GPT2Attention mask.
        if attention_mask is not None and use_rmpad:
            attention_mask2d = attention_mask.max(dim=1)[0]
            del attention_mask
            attention_mask = None  # reduce gpu mem.

        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask_shape = attention_mask.shape
            if len(attention_mask_shape) == 2:  # [bsz, seq] 的情况，多二维，第一维给head，第二维给from_seq，这种情况下无上三角mask
                attention_mask = attention_mask.view(batch_size, -1)
                # We create a 3D attention mask from a 2D tensor mask.
                # Sizes are [batch_size, 1, 1, to_seq_length]
                # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
                # this attention mask is more simple than the triangular masking of causal attention
                # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
                attention_mask = attention_mask[:, None, None, :]
            elif len(attention_mask_shape) == 3:  # [bsz, seq, seq] 的情况，多一维给head，一般这个是训练的情况，提供上三角mask
                attention_mask = attention_mask[:, None, :, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        # head_mask = self.get_head_mask(head_mask, self.config.n_layer)
        head_mask = head_mask = [None] * self.config.n_layer

        if use_rmpad:
            hidden_states, word_idx, seq_lens, max_seqlen = unpad_input(hidden_states, attention_mask2d)
        else:
            word_idx, seq_lens = None, None
        # if token_type_ids is not None:
        #     token_type_embeds = self.wte(token_type_ids) # TODO: 这里好像有点问题，不过现在没用
        #     hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)

        if self.pre_token_proj is not None:  # 映射下
            hidden_states = self.pre_token_proj(hidden_states)

        presents = () if use_cache else None
        all_self_attentions = ()
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logging.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(
                            *inputs,
                            use_cache=use_cache,
                            output_attentions=output_attentions,
                            word_idx=word_idx,
                            seq_lens=seq_lens,
                            use_rmpad=use_rmpad,
                        )

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    None,  # encoder_hidden_states,
                    None,  # encoder_attention_mask,
                )
            else:
                if self.transformer_kernel == "lightseq":
                    outputs = block(
                        hidden_states,
                        attention_mask=attention_mask,
                    )
                else:
                    outputs = block(
                        hidden_states,
                        # layer_past=layer_past,
                        attention_mask=attention_mask,
                        head_mask=head_mask[i],
                        encoder_hidden_states=None,
                        encoder_attention_mask=None,
                        # use_cache=use_cache,
                        output_attentions=output_attentions,
                        word_idx=word_idx,
                        seq_lens=seq_lens,
                        use_rmpad=use_rmpad,
                    )

            if self.transformer_kernel != "deepspeed":
                hidden_states = outputs[0]
                last_activation_norm = outputs[-1]
            else:
                last_activation_norm = None

            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        if self.post_token_proj is not None:  # 映射下
            hidden_states = self.post_token_proj(hidden_states)

        hidden_states = self.ln_f(hidden_states)

        if not use_rmpad:
            hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        all_hidden_states = all_hidden_states + (hidden_states,)

        return {
            "last_hidden_state": hidden_states,
            "word_idx": word_idx,
            "seq_lens": seq_lens,
            # 'past_key_values': presents,
            # 'hidden_states': all_hidden_states,
            # 'attentions': attentions,
            # 'cross_attentions': cross_attentions,
            "last_activation_norm": last_activation_norm,
        }


class GPT2LMHeadModel(nn.Module):
    def __init__(self, config):
        """
        有 head，可以预测loss
        """
        super().__init__()
        # 有点丑，主要是因为 transformer kernel 需要传 batch size
        self.config = config.network
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(self.config.n_embed, self.config.vocab_size, bias=False)

        # TODO: 是否需要tie weight
        if self.config.get("tie_weight", False):
            self.lm_head.weight = self.transformer.wte.weight

        self.PAD_IDX = self.config.get("pad_idx", 2)  # 用来ignore的
        self.loss_fct = CrossEntropyLoss(ignore_index=self.PAD_IDX)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = self.config.get("gradient_checkpointing", False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        use_rmpad: Optional[bool] = False,
        pad_output: Optional[bool] = False,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        # Generate the mask needed for rmpad pkg
        bsz, max_seq_len = input_ids.shape[0], input_ids.shape[1]
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            use_rmpad=use_rmpad,
        )
        hidden_states = transformer_outputs["last_hidden_state"]
        last_activation_norm = transformer_outputs["last_activation_norm"]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)
        if self.gradient_checkpointing and self.training:
            lm_logits = torch.utils.checkpoint.checkpoint(self.lm_head, hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)
        loss = None
        seq_lens = transformer_outputs["seq_lens"]
        word_idx = transformer_outputs["word_idx"]
        if labels is not None:
            if use_rmpad:
                labels = labels.view(-1)[word_idx.long()]
                shift_labels = torch.cat((labels[1:], labels.new_ones((1)) * self.PAD_IDX))
                shift_labels.requires_grad = False
                # first element is 0, ignore it. seq_lens is the cumulative len from 1st seq.
                seq_lens = (seq_lens[1:] - 1).long()
                shift_labels[seq_lens] = self.PAD_IDX
                loss = self.loss_fct(lm_logits, shift_labels)
                # aux_loss = lm_logits.log_softmax(dim=1).square()
                # aux_loss = aux_loss[range(lm_logits.shape[0]), shift_labels]
                # aux_loss[seq_lens] = 0
                # aux_loss = aux_loss.sum() / torch.count_nonzero(aux_loss)
                # loss = loss + 2e-4 * aux_loss
            else:
                # 这里会移位，最后一个token参与encode，但是不参与logits预测
                # label 是去掉第一个token
                # Shift so that tokens < n predict n
                # TODO: 这里没有去掉pad token，虽然loss不算，但可能会增加冗余计算。不过pad应该不多，先不管
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss = self.loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
        if use_rmpad and pad_output:
            lm_logits = pad_input(lm_logits, word_idx, bsz, max_seq_len)

        return {
            "loss": loss,
            "logits": lm_logits,
            "seq_lens": seq_lens,
            # 'past_key_values': transformer_outputs['past_key_values'],
            # 'hidden_states': transformer_outputs['hidden_states'],
            # 'attentions': transformer_outputs['attentions'],
            # 'cross_attentions': transformer_outputs['cross_attentions'],
            "last_activation_norm": transformer_outputs["last_activation_norm"],
        }


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


def load_tf_weights_in_gpt2(model, gpt2_checkpoint_path):
    """Load tf checkpoints in a pytorch model"""
    try:
        import re

        import tensorflow as tf
    except ImportError:
        logging.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(gpt2_checkpoint_path)
    logging.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        if not shape:
            continue
        if not name.startswith("model/"):
            continue
        logging.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array.squeeze())
    logging.info(f"Total variables: {len(names)}")

    for name, array in zip(names, arrays):
        name = name[6:]  # skip "model/"
        name = name.split("/")
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                scope_names = re.split(r"(\d+)", m_name)
            else:
                scope_names = [m_name]
            logging.info(f"Converting m_name: {m_name} to scope_names: {scope_names}")
            if scope_names[0] == "w" or scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "wpe" or scope_names[0] == "wte":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, scope_names[0])
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        try:
            assert (
                pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logging.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    return model
