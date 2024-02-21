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
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from mariana.models.gpt2 import GPT2Model
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from .xperf_training import FTFlashAttention, FTLayerNorm, FTLinear, FTLinearWeightTransposed, pad_input, unpad_input


class GPT2LMModelwClassificationHeadRMPAD(nn.Module):
    def __init__(self, config):
        """
        有 head，可以预测loss
        """
        super().__init__()
        self.config = config.network  # 有点丑，主要是因为 transformer kernel 需要传 batch size
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(self.config.n_embed, self.config.vocab_size, bias=False)

        # binary classification, biase = False/True, Tune later
        print("classification_head_bias: {}".format(self.config.classification_head_bias))
        self.classification_head = nn.Linear(
            self.config.n_embed,
            self.config.num_labels,
            bias=self.config.classification_head_bias,
        )

        # TODO: 是否需要tie weight
        if self.config.get("tie_weight", False):
            self.lm_head.weight = self.transformer.wte.weight

        self.PAD_IDX = self.config.get("pad_idx", -100)  # 用来ignore的
        print("PAD_IDX is: {}".format(self.PAD_IDX))
        self.loss_fct = CrossEntropyLoss(ignore_index=self.PAD_IDX)
        self.loss_fct_classification = CrossEntropyLoss(ignore_index=self.PAD_IDX)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # def parallelize(self, device_map=None):
        #     self.device_map = (
        #         get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
        #         if device_map is None
        #         else device_map
        #     )
        #     assert_device_map(self.device_map, len(self.transformer.h))
        #     self.transformer.parallelize(self.device_map)
        #     self.lm_head = self.lm_head.to(self.transformer.first_device)
        #     self.model_parallel = True

        # def deparallelize(self):
        #     self.transformer.deparallelize()
        #     self.transformer = self.transformer.to("cpu")
        #     self.lm_head = self.lm_head.to("cpu")
        #     self.model_parallel = False
        torch.cuda.empty_cache()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        classification_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        actual_seq_length=None,
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

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        if not use_rmpad:
            # print("actual_seq_length: {}".format(actual_seq_length))
            actual_seq_length = actual_seq_length.unsqueeze(-1) - 1  # bz * 1
            hidden_size = hidden_states.size(dim=-1)
            # print("hidden_size: {}".format(hidden_size))
            indices = actual_seq_length.repeat(1, hidden_size)  # bz * hidden_size
            indices = indices.unsqueeze(1)  # bz * 1 * hidden_size
            # print("hidden_states_size: {}, indices_size: {}".format(hidden_states.size(), indices.size()))

            eos_output = torch.gather(hidden_states, 1, indices)  # bz * 1 * hidden_size
            eos_output = eos_output.squeeze(1)
            # print("eos_output: {} with size: {}".format(eos_output, eos_output.size()))
            # print("size of hidden_states[:,-1,:].squeeze(1): {}".format(hidden_states[:,-1,:].squeeze(1).size()))
        else:
            seq_lens = transformer_outputs["seq_lens"][1:]
            # seq_lens: tensor([101, 243, 307, 436], device='cuda:3', dtype=torch.int32)
            eos_output = torch.index_select(hidden_states, 0, seq_lens - 1)

        # option #1: last hidden
        classification_logits = self.classification_head(eos_output)

        # classification_logits = self.classification_head(hidden_states[:,-1,:].squeeze(1))

        # option #2: mean
        # classification_logits = self.classification_head(torch.mean(hidden_states, 1, False))

        lm_loss, classification_loss = 0.0, 0.0

        # if labels is not None:
        #     # 这里会移位，最后一个token参与encode，但是不参与logits预测
        #     # label 是去掉第一个token
        #     # Shift so that tokens < n predict n
        #     # TODO: 这里没有去掉pad token，虽然loss不算，但可能会增加冗余计算。不过pad应该不多，先不管
        #     shift_logits = lm_logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()
        #     # Flatten the tokens
        #     lm_loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        seq_lens = transformer_outputs["seq_lens"]
        word_idx = transformer_outputs["word_idx"]
        if labels is not None:
            if use_rmpad:
                labels = labels.view(-1)[word_idx.long()]
                shift_labels = torch.cat((labels[1:], labels.new_ones((1)) * self.PAD_IDX))
                shift_labels.requires_grad = False
                seq_lens = (
                    seq_lens[1:] - 1
                ).long()  # first element is 0, ignore it. seq_lens is the cumulative len from 1st seq.
                shift_labels[seq_lens] = self.PAD_IDX
                # print("seq_lens from transformer_outputs: {}, actual_seq_lenght from dataloader: {}, pad_idx: {}".format(seq_lens, actual_seq_length, self.PAD_IDX))
                lm_loss = self.loss_fct(lm_logits, shift_labels)
            else:
                # 这里会移位，最后一个token参与encode，但是不参与logits预测
                # label 是去掉第一个token
                # Shift so that tokens < n predict n
                # TODO: 这里没有去掉pad token，虽然loss不算，但可能会增加冗余计算。不过pad应该不多，先不管
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                lm_loss = self.loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
        if use_rmpad and pad_output:
            lm_logits = pad_input(lm_logits, word_idx, bsz, max_seq_len)

        if classification_labels is not None:
            # print("classification_labels: ")
            # print(classification_labels)
            # print("classification_logits")
            # print(classification_logits)
            # print("classfication_logits_size")
            # print(classification_logits.size())
            classification_loss = self.loss_fct_classification(
                classification_logits, classification_labels.long().view(-1)
            )

        loss = self.config.lm_loss_weight * lm_loss + classification_loss

        return {
            "loss": loss,
            "lm_loss": lm_loss,
            "classification_loss": classification_loss,
            "logits": lm_logits,
            "classification_logits": classification_logits,
            "seq_lens": seq_lens,
            #'past_key_values': transformer_outputs['past_key_values'],
            #'hidden_states': transformer_outputs['hidden_states'],
            #'attentions': transformer_outputs['attentions'],
            #'cross_attentions': transformer_outputs['cross_attentions'],
        }


class GPT2LMModelwClassificationHead(nn.Module):
    def __init__(self, config):
        """
        有 head，可以预测loss
        """
        super().__init__()
        self.config = config.network  # 有点丑，主要是因为 transformer kernel 需要传 batch size
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(self.config.n_embed, self.config.vocab_size, bias=False)

        # binary classification, biase = False/True, Tune later
        print("classification_head_bias: {}".format(self.config.classification_head_bias))
        self.classification_head = nn.Linear(
            self.config.n_embed,
            self.config.num_labels,
            bias=self.config.classification_head_bias,
        )

        # TODO: 是否需要tie weight
        if self.config.get("tie_weight", False):
            self.lm_head.weight = self.transformer.wte.weight

        self.PAD_IDX = self.config.get("pad_idx", -100)  # 用来ignore的
        print("PAD_IDX is: {}".format(self.PAD_IDX))
        self.loss_fct = CrossEntropyLoss(ignore_index=self.PAD_IDX)
        self.loss_fct_classification = CrossEntropyLoss(ignore_index=self.PAD_IDX)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # def parallelize(self, device_map=None):
        #     self.device_map = (
        #         get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
        #         if device_map is None
        #         else device_map
        #     )
        #     assert_device_map(self.device_map, len(self.transformer.h))
        #     self.transformer.parallelize(self.device_map)
        #     self.lm_head = self.lm_head.to(self.transformer.first_device)
        #     self.model_parallel = True

        # def deparallelize(self):
        #     self.transformer.deparallelize()
        #     self.transformer = self.transformer.to("cpu")
        #     self.lm_head = self.lm_head.to("cpu")
        #     self.model_parallel = False
        torch.cuda.empty_cache()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        classification_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        actual_seq_length=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = transformer_outputs["last_hidden_state"]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        actual_seq_length = actual_seq_length.unsqueeze(-1) - 1  # bz * 1
        # print("actual_seq_length: {} with size: {}".format(actual_seq_length, actual_seq_length.size()))
        hidden_size = hidden_states.size(dim=-1)
        # print("hidden_states_size: {}".format(hidden_states.size()))
        indices = actual_seq_length.repeat(1, hidden_size)  # bz * hidden_size
        indices = indices.unsqueeze(1)  # bz * 1 * hidden_size
        # print("hidden_states_size: {}, indices_size: {}".format(hidden_states.size(), indices.size()))
        eos_output = torch.gather(hidden_states, 1, indices)  # bz * 1 * hidden_size
        # print("eos_output: {} with size: {}".format(eos_output, eos_output.size()))
        # print("size of hidden_states[:,-1,:].squeeze(1): {}".format(hidden_states[:,-1,:].squeeze(1).size()))

        # option #1: last hidden
        classification_logits = self.classification_head(eos_output.squeeze(1))

        # classification_logits = self.classification_head(hidden_states[:,-1,:].squeeze(1))

        # option #2: mean
        # classification_logits = self.classification_head(torch.mean(hidden_states, 1, False))

        lm_loss, classification_loss = 0.0, 0.0

        if labels is not None:
            # 这里会移位，最后一个token参与encode，但是不参与logits预测
            # label 是去掉第一个token
            # Shift so that tokens < n predict n
            # TODO: 这里没有去掉pad token，虽然loss不算，但可能会增加冗余计算。不过pad应该不多，先不管
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            lm_loss = self.loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        if classification_labels is not None:
            # print("classification_labels: ")
            # print(classification_labels)
            # print("classification_logits")
            # print(classification_logits)
            # print("classfication_logits_size")
            # print(classification_logits.size())
            classification_loss = self.loss_fct_classification(
                classification_logits, classification_labels.long().view(-1)
            )

        loss = self.config.lm_loss_weight * lm_loss + classification_loss

        return {
            "loss": loss,
            "lm_loss": lm_loss,
            "classification_loss": classification_loss,
            "logits": lm_logits,
            "classification_logits": classification_logits,
            #'past_key_values': transformer_outputs['past_key_values'],
            #'hidden_states': transformer_outputs['hidden_states'],
            #'attentions': transformer_outputs['attentions'],
            #'cross_attentions': transformer_outputs['cross_attentions'],
        }
