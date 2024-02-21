"""https://arxiv.org/abs/2006.03654
Standalone, all-in-one code copied from ptx deberta implementation:
https://code.byted.org/nlp/ptx/blob/master/ptx/model/deberta/model.py
"""
from functools import partial
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ....utils import load_pretrained_model_weights, logging
from ...components.bert_components import (
    BertEmbedding,
    BertEmbeddingPinyin,
    BertPooler,
    BertPreTrainingHeads,
    init_weights,
)
from ...components.disentangled_attention import DAConfig, DisentangledMHA, build_relative_position
from ...components.transformer_encoder_decoder import (
    BareTransformerEncoder,
    BareTransformerEncoderLayer,
    TransformerEncoderLayerOutput,
    TransformerEncoderOutput,
    _expand_mask,
)
from ...modeling_utils import ModelBase


def download_weights():
    ...


def get_configs():
    ...


"""Deberta model"""

__all__ = ["DebertaModel"]


logger = logging.get_logger(__name__)


def gather_positions(sequence: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """
    Args:
        sequence: tensor (batch_size, seq_len, dim)
        positions: tensor (batch_size, number_of_tokens_to_be_masked)

    Returns:
        tensor (batch_size, number_of_tokens_to_be_masked, dim)
    """

    batch_size, seq_len, dim = sequence.shape
    position_shift = (seq_len * torch.arange(batch_size, device=sequence.device)).unsqueeze(-1)
    flat_positions = torch.reshape(positions + position_shift, [-1]).long()
    flat_sequence = torch.reshape(sequence, [batch_size * seq_len, dim])
    gathered = flat_sequence.index_select(0, flat_positions)
    return torch.reshape(gathered, [batch_size, -1, dim])


class CategoricalAccuracy:
    """
    Categorical Top-K accuracy. Assumes integer labels, with
    each item to be classified having a single correct class.
    Tie break enables equal distribution of scores among the
    classes with same maximum predicted scores.
    """

    def __init__(self, top_k: int = 1, tie_break: bool = False) -> None:
        if top_k > 1 and tie_break:
            raise Exception("Tie break in Categorical Accuracy can be done only for maximum (top_k = 1)")
        if top_k <= 0:
            raise Exception("top_k passed to Categorical Accuracy must be > 0")
        self._top_k = top_k
        self._tie_break = tie_break
        self.correct_count = 0.0
        self.total_count = 0.0
        self.local_value = 0.0

    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            predictions : `torch.Tensor`, required.
                A tensor of predictions of shape (batch_size, ..., num_classes).
            gold_labels : `torch.Tensor`, required.
                A tensor of integer class label of shape (batch_size, ...). It must be the same
                shape as the `predictions` tensor without the `num_classes` dimension.
            mask : `torch.Tensor`, optional (default = None).
                A masking tensor the same size as `gold_labels`.
        """
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)

        predictions_within_classes = True
        if predictions.dim() - gold_labels.dim() == 1:
            num_classes = predictions.size(-1)
            if (gold_labels >= num_classes).any():
                raise Exception(
                    "A gold label passed to Categorical Accuracy contains an id >= {}, "
                    "the number of classes.".format(num_classes)
                )
            predictions = predictions.view((-1, num_classes)).float()
        else:
            assert self._top_k == 1, "`top_k` should be 1 if `predictions` has no `num_classes` dimension."
            predictions_within_classes = False
            predictions = predictions.view(-1).float()

        gold_labels = gold_labels.view(-1).long()
        if not self._tie_break:
            # Top K indexes of the predictions (or fewer, if there aren't K of them).
            # Special case topk == 1, because it's common and .max() is much faster than .topk().
            if self._top_k == 1:
                top_k = (
                    predictions.max(-1)[1].unsqueeze(-1) if predictions_within_classes else predictions.unsqueeze(-1)
                )
            else:
                top_k = predictions.topk(min(self._top_k, predictions.shape[-1]), -1)[1]

            # This is of shape (batch_size, ..., top_k).
            correct = top_k.eq(gold_labels.unsqueeze(-1)).float()
        else:
            # TODO
            assert predictions_within_classes, "`tie_break` requires `predictions` with `num_classes` dimension."

            # prediction is correct if gold label falls on any of the max scores. distribute score by tie_counts
            max_predictions = predictions.max(-1)[0]
            max_predictions_mask = predictions.eq(max_predictions.unsqueeze(-1))
            # max_predictions_mask is (rows X num_classes) and gold_labels is (batch_size)
            # ith entry in gold_labels points to index (0-num_classes) for ith row in max_predictions
            # For each row check if index pointed by gold_label is was 1 or not (among max scored classes)
            correct = max_predictions_mask[
                torch.arange(gold_labels.numel(), device=gold_labels.device).long(),
                gold_labels,
            ].float()
            tie_counts = max_predictions_mask.sum(-1)
            correct /= tie_counts.float()
            correct.unsqueeze_(-1)

        total_count_add = 0
        if mask is not None:
            correct *= mask.view(-1, 1).float()
            total_count_add = mask.sum()
        else:
            total_count_add = gold_labels.numel()
        correct_count_add = correct.sum()
        if total_count_add > 1e-12:
            self.local_value = float(correct_count_add) / float(total_count_add)
        else:
            self.local_value = 0.0
        self.total_count += total_count_add
        self.correct_count += correct_count_add

    def get_metric(self, reset: bool = False, use_local: bool = False):
        """
        Returns:
            The accumulated accuracy.
        """
        if not use_local:
            if self.total_count > 1e-12:
                accuracy = float(self.correct_count) / float(self.total_count)
            else:
                accuracy = 0.0
        else:
            accuracy = self.local_value
        if reset:
            self.reset()
        return accuracy

    def reset(self):
        self.correct_count = 0.0
        self.total_count = 0.0
        self.local_value = 0.0


"""Deberta models"""


class DebertaEncoderLayer(BareTransformerEncoderLayer):
    attention_class = DisentangledMHA

    def __init__(self, config: DAConfig, order=-1, **kwargs):
        if isinstance(config, dict):
            config = DAConfig(**config)
        super().__init__(config, order=order)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor,
        relative_pos: torch.Tensor,
        relative_pos_embed: torch.Tensor,
        q_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # residual = hidden_states
        residual = hidden_states if q_state is None else q_state
        self_attn_output = self.attn(
            hidden_states,
            attn_mask=attn_mask,
            # The only diff from original Transformer encoder layer
            relative_pos=relative_pos,
            relative_pos_embed=relative_pos_embed,
            q_state=q_state,
        )
        if not self.config.obey_other_attn_output:
            hidden_states = self.proj(self_attn_output)
        else:
            hidden_states = self.proj(self_attn_output.attn_outputs)
        hidden_states = self.dropout1(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.norm1(hidden_states)

        # Position-wise Feed-Forward Networks
        residual = hidden_states
        if self._use_moe:
            hidden_states, l_aux, _ = self.pwff(hidden_states)
            if self.training:
                self.l_aux = l_aux
                self.z_loss = self.pwff.get_z_loss()
        else:
            hidden_states = self.pwff(hidden_states)
        if self.training and getattr(self.attn, "l_aux", False):
            self.l_aux_attn = self.attn.l_aux
        hidden_states = self.dropout2(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.norm2(hidden_states)

        if not self.config.obey_other_attn_output:
            return hidden_states
        return TransformerEncoderLayerOutput(
            hidden_states=hidden_states,
            attn_probs=self_attn_output.attn_probs,
            attn_weights=self_attn_output.attn_weights,
            q_states=self_attn_output.q_states,
            k_states=self_attn_output.k_states,
            v_states=self_attn_output.v_states,
            attn_bias=self_attn_output.attn_bias,
        )


class DebertaEncoder(BareTransformerEncoder):
    layer_class = DebertaEncoderLayer

    def __init__(self, config: DAConfig):
        if isinstance(config, dict):
            config = DAConfig(**config)
        super().__init__(config)

        self.max_relative_positions = config.max_relative_positions
        self.position_buckets = config.position_buckets
        if self.position_buckets > 0:
            pos_ebd_size = self.position_buckets * 2
        else:
            pos_ebd_size = self.max_relative_positions * 2
        self.rel_embeddings = nn.Embedding(pos_ebd_size, self.dim)
        self.l_aux = list()
        self.z_loss = list()
        self.l_aux_attn = list()

    def forward(self, hidden_states, attn_mask=None) -> List[torch.Tensor]:
        attn_mask_expanded = _expand_mask(attn_mask, hidden_states.dtype) if attn_mask.dim() != 4 else attn_mask

        relative_pos = build_relative_position(
            hidden_states.size(-2),
            hidden_states.size(-2),
            bucket_size=self.position_buckets,
            max_position=self.max_relative_positions,
            device=hidden_states.device,
        )

        all_hidden_states = []
        if 0 in self.config.return_layers:
            all_hidden_states.append(hidden_states)
        all_attn_weights = []
        all_attn_probs = []
        all_q_states = []
        all_k_states = []
        all_v_states = []

        l_aux = list()
        z_loss = list()
        l_aux_attn = list()
        for idx, block in enumerate(self.blocks):
            layer_output = block(
                hidden_states,
                attn_mask_expanded,
                relative_pos,
                self.rel_embeddings.weight,
            )
            if not self.config.obey_other_attn_output:
                hidden_states = layer_output
            else:
                hidden_states = layer_output.hidden_states
            if (idx + 1) in self.config.return_layers:
                all_hidden_states.append(hidden_states)
                if self.config.obey_other_attn_output:
                    all_attn_weights.append(layer_output.attn_weights)
                    all_attn_probs.append(layer_output.attn_probs)
                    all_q_states.append(layer_output.q_states)
                    all_k_states.append(layer_output.k_states)
                    all_v_states.append(layer_output.v_states)
            if self.training and getattr(block, "_use_moe", False):
                l_aux.append(block.l_aux)
                z_loss.append(block.z_loss)
            if self.training and getattr(block.attn, "use_moe", False):
                l_aux_attn.append(block.l_aux_attn)
        if self.training:
            self.l_aux = l_aux
            self.z_loss = z_loss
            self.l_aux_attn = l_aux_attn

        if not self.config.obey_other_attn_output:
            return hidden_states, all_hidden_states, relative_pos
        return TransformerEncoderOutput(
            last_hidden_state=hidden_states,
            all_hidden_states=all_hidden_states,
            all_attn_probs=all_attn_probs,
            all_attn_weights=all_attn_weights,
            all_q_states=all_q_states,
            all_k_states=all_k_states,
            all_v_states=all_v_states,
        )


class DebertaBare(nn.Module):
    "The bare DeBERTa model, outputting raw hidden states without any specific head."

    def __init__(
        self,
        vocab_size: int,
        dim: int = 768,
        dim_ff: int = 3072,
        dim_shrink: int = 0,
        n_segments: int = 2,
        p_drop_hidden: float = 0.1,
        n_heads: int = 12,
        n_layers: int = 12,
        initializer_range: float = 0.02,
        layernorm_type: str = "default",
        layernorm_fp16: bool = False,
        layer_norm_eps: float = 1e-12,
        p_drop_attn: float = 0.1,
        embedding_dim: int = None,
        token_embedding_dim: int = None,
        embedding_dropout: float = None,
        act: str = "gelu",
        pool: bool = True,
        padding_index: int = 0,
        attention_clamp_inf: bool = False,
        max_relative_positions: int = 512,
        extra_da_transformer_config: Optional[Dict] = None,
        omit_other_output: bool = False,
        use_fast: bool = False,
        **kwargs,
    ):
        """
        Args:
            option: See `DebertaEncoder` and `BertEmbedding`
        """
        super().__init__()
        if use_fast:
            raise NotImplementedError("use_fast version of deberta is not supported in titan.")
        embedding_dim = embedding_dim or dim
        self.initializer_range = initializer_range
        self.vocab_size = vocab_size

        self.embedding = BertEmbedding(
            dim=embedding_dim,
            vocab_size=self.vocab_size,
            n_segments=n_segments,
            max_len=0,
            p_drop_hidden=p_drop_hidden if embedding_dropout is None else embedding_dropout,
            layer_norm_eps=layer_norm_eps,
            layernorm_type=layernorm_type,
            token_embedding_dim=token_embedding_dim,
        )
        if layernorm_fp16:
            self.embedding.norm._simply_cast = True

        self.proj_embedding_hidden = None
        if embedding_dim != dim:
            self.proj_embedding_hidden = torch.nn.Linear(embedding_dim, dim)

        self._omit_other_output = omit_other_output
        self.extra_da_transformer_config = extra_da_transformer_config
        self.da_config = DAConfig(
            n_layers,
            dim,
            n_heads,
            dim_ff,
            act=act,
            layernorm_type=layernorm_type,
            p_drop_hidden=p_drop_hidden,
            p_drop_attn=p_drop_attn,
            return_layers=list(range(n_layers + 1)) if not self._omit_other_output else [],
            clamp_inf_nan=attention_clamp_inf,
            layer_norm_eps=layer_norm_eps,
            max_relative_positions=max_relative_positions,
            **(extra_da_transformer_config or {}),
        )
        self.encoder = DebertaEncoder(self.da_config)
        self._use_moe = self.encoder.config.use_moe
        self._use_moe_attn = self.encoder.config.use_moe_attn

        self.pooler = BertPooler(dict(dim=dim)) if pool else None

        self.proj_shrink_dim = nn.Linear(dim, dim_shrink) if dim_shrink and self.pooler else None

        self.apply(partial(init_weights, initializer_range=self.initializer_range))

        self.padding_index = padding_index

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        output_pooled=False,
        output_rel_pos=False,
        position_ids=None,  # Useless, for api compat
        output_qkv=False,
    ):
        if attention_mask is None:
            attention_mask = (input_ids != self.padding_index).to(
                dtype=self.embedding.token_embedder_tokens.weight.dtype
            )

        embedding_output = self.embedding(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            # position_ids=position_ids,
            mask=attention_mask,
        )

        if self.proj_embedding_hidden is not None:
            embedding_output = self.proj_embedding_hidden(embedding_output)

        encoder_out = self.encoder(embedding_output, attention_mask)
        if not self.da_config.obey_other_attn_output:
            (
                sequence_output,
                encoder_all_hidden_states,
                relative_pos,
            ) = encoder_out
        else:
            sequence_output = encoder_out.last_hidden_state
            encoder_all_hidden_states = encoder_out.all_hidden_states
            relative_pos = None  # TODO: sure?

        pooled_output = None
        shrinked_output = None
        if (self.pooler is not None) and output_pooled:
            pooled_output = self.pooler(sequence_output)
            if self.proj_shrink_dim:
                shrinked_output = self.proj_shrink_dim(pooled_output)

        ret = {"sequence_output": sequence_output}

        if self._use_moe:
            loss = 0
            for i in range(len(self.encoder.l_aux)):
                loss += self.extra_da_transformer_config.get("moe_l_aux_factor", 0.01) * self.encoder.l_aux[i]
                loss += self.extra_da_transformer_config.get("moe_z_loss_factor", 0.0) * self.encoder.z_loss[i]
            ret["loss"] = loss
        if self._use_moe_attn:
            loss = ret.get("loss", 0)
            for i in range(len(self.encoder.l_aux_attn)):
                loss += (
                    self.extra_da_transformer_config.get("moe_l_aux_factor_attn", 0.01) * self.encoder.l_aux_attn[i]
                )
            ret["loss"] = loss

        if output_qkv and self.da_config.obey_other_attn_output:
            ret["all_q_states"] = encoder_out.all_q_states
            ret["all_k_states"] = encoder_out.all_k_states
            ret["all_v_states"] = encoder_out.all_v_states
        if self._omit_other_output:
            if output_pooled:
                ret["pooled_output"] = pooled_output
            else:
                ret["pooled_output"] = None
            ret["shrinked_output"] = shrinked_output
            if output_rel_pos:
                if relative_pos is None:  # In case of FT, which does not return rel pos
                    # print('Recreating relative_pos')
                    relative_pos = build_relative_position(
                        embedding_output.size(-2),
                        embedding_output.size(-2),
                        bucket_size=self.encoder.position_buckets,
                        max_position=self.encoder.max_relative_positions,
                        device=embedding_output.device,
                    )
                ret["relative_pos"] = relative_pos
            return ret
        else:
            ret.update(
                {
                    "pooled_output": pooled_output,
                    "relative_pos": relative_pos,
                    "mask": attention_mask,
                    "embedding": embedding_output,
                    "hidden": encoder_all_hidden_states,
                    "attention": None,
                }
            )
            return ret


class DebertaEMD(nn.Module):
    def __init__(
        self,
        config: DAConfig,
        num_emd_groups: int = 1,
        emd_group_repeat: int = 2,
        use_fast: bool = False,
    ):
        super().__init__()
        assert not use_fast
        self.blocks = nn.ModuleList([DebertaEncoderLayer(config) for _ in range(num_emd_groups)])
        self.group_repeat = emd_group_repeat

    def forward(self, i, h, mask, relative_pos=None, rel_embedding=None):
        attn_mask_expanded = _expand_mask(mask, h.dtype) if mask.dim() != 4 else mask
        for m, block in enumerate(self.blocks):
            for n in range(self.group_repeat):
                i = block(
                    h,
                    attn_mask_expanded,
                    relative_pos,
                    rel_embedding,
                    q_state=i,
                )
        return i


class DebertaBarePinyin(nn.Module):
    "The bare DeBERTa model, outputting raw hidden states without any specific head."

    def __init__(
        self,
        vocab_size: int,
        pinyin_vocab_size: int,
        dim: int = 768,
        dim_ff: int = 3072,
        n_segments: int = 2,
        p_drop_hidden: float = 0.1,
        n_heads: int = 12,
        n_layers: int = 12,
        initializer_range: float = 0.02,
        layernorm_type: str = "default",
        layernorm_fp16: bool = False,
        layer_norm_eps: float = 1e-12,
        p_drop_attn: float = 0.1,
        embedding_dim: int = None,
        token_embedding_dim: int = None,
        embedding_dropout: float = None,
        act: str = "gelu",
        pool: bool = True,
        padding_index: int = 0,
        attention_clamp_inf: bool = False,
        max_relative_positions: int = 512,
        extra_da_transformer_config: Optional[Dict] = None,
        omit_other_output: bool = False,
        use_fast: bool = False,
        **kwargs,
    ):
        """
        Args:
            option: See `DebertaEncoder` and `BertEmbedding`
        """
        super().__init__()
        if use_fast:
            raise NotImplementedError("use_fast version of deberta is not supported in titan.")
        embedding_dim = embedding_dim or dim
        self.initializer_range = initializer_range
        self.vocab_size = vocab_size

        self.embedding = BertEmbeddingPinyin(
            dim=embedding_dim,
            vocab_size=self.vocab_size,
            pinyin_vocab_size=pinyin_vocab_size,
            n_segments=n_segments,
            max_len=0,
            p_drop_hidden=p_drop_hidden if embedding_dropout is None else embedding_dropout,
            layer_norm_eps=layer_norm_eps,
            layernorm_type=layernorm_type,
            token_embedding_dim=token_embedding_dim,
        )
        if layernorm_fp16:
            self.embedding.norm._simply_cast = True

        self.proj_embedding_hidden = None
        if embedding_dim != dim:
            self.proj_embedding_hidden = torch.nn.Linear(embedding_dim, dim)

        self._omit_other_output = omit_other_output
        self.extra_da_transformer_config = extra_da_transformer_config
        self.da_config = DAConfig(
            n_layers,
            dim,
            n_heads,
            dim_ff,
            act=act,
            layernorm_type=layernorm_type,
            p_drop_hidden=p_drop_hidden,
            p_drop_attn=p_drop_attn,
            return_layers=list(range(n_layers + 1)) if not self._omit_other_output else [],
            clamp_inf_nan=attention_clamp_inf,
            layer_norm_eps=layer_norm_eps,
            max_relative_positions=max_relative_positions,
            **(extra_da_transformer_config or {}),
        )
        self.encoder = DebertaEncoder(self.da_config)
        self._use_moe = self.encoder.config.use_moe
        self._use_moe_attn = self.encoder.config.use_moe_attn

        self.pooler = BertPooler(dict(dim=dim)) if pool else None

        self.apply(partial(init_weights, initializer_range=self.initializer_range))

        self.padding_index = padding_index

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        output_pooled=False,
        output_rel_pos=False,
        position_ids=None,  # Useless, for api compat
        output_qkv=False,
        pinyin_ids=None,
    ):
        if attention_mask is None:
            attention_mask = (input_ids != self.padding_index).to(
                dtype=self.embedding.token_embedder_tokens.weight.dtype
            )

        embedding_output = self.embedding(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            # position_ids=position_ids,
            mask=attention_mask,
            pinyin_ids=pinyin_ids,
        )

        if self.proj_embedding_hidden is not None:
            embedding_output = self.proj_embedding_hidden(embedding_output)

        encoder_out = self.encoder(embedding_output, attention_mask)
        if not self.da_config.obey_other_attn_output:
            (
                sequence_output,
                encoder_all_hidden_states,
                relative_pos,
            ) = encoder_out
        else:
            sequence_output = encoder_out.last_hidden_state
            encoder_all_hidden_states = encoder_out.all_hidden_states
            relative_pos = None  # TODO: sure?

        pooled_output = None
        if (self.pooler is not None) and output_pooled:
            pooled_output = self.pooler(sequence_output)

        ret = {"sequence_output": sequence_output}

        if self._use_moe:
            loss = 0
            for i in range(len(self.encoder.l_aux)):
                loss += self.extra_da_transformer_config.get("moe_l_aux_factor", 0.01) * self.encoder.l_aux[i]
                loss += self.extra_da_transformer_config.get("moe_z_loss_factor", 0.0) * self.encoder.z_loss[i]
            ret["loss"] = loss
        if self._use_moe_attn:
            loss = ret.get("loss", 0)
            for i in range(len(self.encoder.l_aux_attn)):
                loss += (
                    self.extra_da_transformer_config.get("moe_l_aux_factor_attn", 0.01) * self.encoder.l_aux_attn[i]
                )
            ret["loss"] = loss

        if output_qkv and self.da_config.obey_other_attn_output:
            ret["all_q_states"] = encoder_out.all_q_states
            ret["all_k_states"] = encoder_out.all_k_states
            ret["all_v_states"] = encoder_out.all_v_states
        if self._omit_other_output:
            if output_pooled:
                ret["pooled_output"] = pooled_output
            else:
                ret["pooled_output"] = None
            if output_rel_pos:
                if relative_pos is None:  # In case of FT, which does not return rel pos
                    # print('Recreating relative_pos')
                    relative_pos = build_relative_position(
                        embedding_output.size(-2),
                        embedding_output.size(-2),
                        bucket_size=self.encoder.position_buckets,
                        max_position=self.encoder.max_relative_positions,
                        device=embedding_output.device,
                    )
                ret["relative_pos"] = relative_pos
            return ret
        else:
            ret.update(
                {
                    "pooled_output": pooled_output,
                    "relative_pos": relative_pos,
                    "mask": attention_mask,
                    "embedding": embedding_output,
                    "hidden": encoder_all_hidden_states,
                    "attention": None,
                }
            )
            return ret


class DebertaModel(DebertaBare, ModelBase):
    def __init__(
        self,
        max_len: int = 512,
        abs_pos_embedding: bool = False,
        ignore_index: int = -1,
        calc_mlm_accuracy: bool = True,
        tie_embedding: bool = True,
        use_emd: bool = False,
        num_emd_groups: int = 1,
        emd_group_repeat: int = 2,
        layernorm_fp16: bool = False,
        use_fast: bool = False,
        head_layernorm_type: str = "default",
        omit_other_output: bool = False,
        **option,
    ):
        super().__init__(
            use_fast=use_fast,
            layernorm_fp16=layernorm_fp16,
            omit_other_output=omit_other_output,
            **option,
        )

        if abs_pos_embedding:
            self.ape = nn.Embedding(max_len, option["dim"])
        else:
            self.ape = None
        self.cls = BertPreTrainingHeads(
            dict(
                dim=option["dim"],
                embedding_dim=option["embedding_dim"],
                layer_norm_eps=option["layer_norm_eps"],
                vocab_size=option["vocab_size"],
                act=option["act"],
                layernorm_type=head_layernorm_type,
            )
        )
        if layernorm_fp16:
            self.cls.predictions.layer_norm._simply_cast = True
        if tie_embedding:
            self._tie_weights()

        self._omit_other_output = omit_other_output
        self._calc_mlm_accuracy = calc_mlm_accuracy and not self._omit_other_output
        self._use_moe = self.encoder.config.use_moe
        self._use_moe_attn = self.encoder.config.use_moe_attn
        self.mlm_accuracy = self._calc_mlm_accuracy and CategoricalAccuracy()

        self.ignore_index = ignore_index
        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.nsp_loss_function = torch.nn.CrossEntropyLoss()

        self.local_metrics = {}

        self.use_emd = use_emd
        if use_emd:
            self.emd = DebertaEMD(
                self.da_config,
                num_emd_groups=num_emd_groups,
                emd_group_repeat=emd_group_repeat,
                use_fast=use_fast,
            )

        self.apply(partial(init_weights, initializer_range=self.initializer_range))

    def load_pretrained_weights(self, weight_file_path, **kwargs):
        load_pretrained_model_weights(self, weight_file_path, rm_prefix="deberta.", **kwargs)
        ...

    def _tie_weights(self):
        self.cls.predictions.decoder.weight = self.embedding.token_embedder_tokens.weight

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        global_metrics = {}
        if self._calc_mlm_accuracy:
            global_metrics["mlm_accuracy"] = self.mlm_accuracy.get_metric(reset)
        global_metrics.update(self.local_metrics)
        return global_metrics

    def _update_local_metrics(self, mlm_logits, mlm_labels):
        if self._calc_mlm_accuracy:
            total_count, correct_count = float(self.mlm_accuracy.total_count), float(self.mlm_accuracy.correct_count)
            mlm_positions = torch.nonzero(mlm_labels != self.ignore_index, as_tuple=False).view(-1)
            self.mlm_accuracy(mlm_logits[mlm_positions], mlm_labels[mlm_positions])
            local_total_count = float(self.mlm_accuracy.total_count) - total_count
            local_correct_count = float(self.mlm_accuracy.correct_count) - correct_count
            local_accuracy = 0.0 if local_total_count == 0 else (float(local_correct_count) / float(local_total_count))
            self.local_metrics.update(
                {
                    "local_mlm_total_count": local_total_count,
                    "local_mlm_correct_count": local_correct_count,
                    "local_mlm_accuracy": local_accuracy,
                }
            )

    def forward(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
        masked_tokens=None,
        sentence_label=None,
        masked_lm_positions=None,
        masked_lm_ids=None,
        output_pooled=True,
    ):
        mask = (
            attention_mask
            if attention_mask is not None
            else torch.where(
                input_ids
                != torch.tensor(
                    self.padding_index,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                ),
                torch.tensor(1, dtype=input_ids.dtype, device=input_ids.device),
                torch.tensor(0, dtype=input_ids.dtype, device=input_ids.device),
            )
        )

        output = super().forward(
            input_ids=input_ids,
            attention_mask=mask,
            token_type_ids=token_type_ids,
            # position_ids=position_ids,
            output_pooled=output_pooled or sentence_label is not None,
            output_rel_pos=self.use_emd,
        )

        sequence_output = output["sequence_output"]
        pooled_output = output["pooled_output"]
        # shrinked_output = output["shrinked_output"]

        encoder_last_seq_output = sequence_output

        if self.use_emd:
            if self.ape is not None and position_ids is not None:
                abs_pos_embeddings = self.ape(position_ids.long())
                sequence_output = self.emd(
                    abs_pos_embeddings + sequence_output,
                    sequence_output,
                    mask,
                    relative_pos=output["relative_pos"],
                    rel_embedding=self.encoder.rel_embeddings.weight,
                )
            else:
                # TODO: fix
                sequence_output = self.emd(sequence_output, sequence_output)

        decoder_last_seq_output = sequence_output
        # print('decoder_last_seq_output', sequence_output)

        if masked_tokens is None and sentence_label is None and masked_lm_positions is None and masked_lm_ids is None:
            return {
                "sequence_output": decoder_last_seq_output,
                "pooled_output": pooled_output,
                # "shrinked_output": shrinked_output,
            }
            # return decoder_last_seq_output, pooled_output

        # Shrink `sequence_output` and `masked_tokens` according to `masked_lm_positions` and `masked_lm_ids`
        positioned = masked_lm_positions is not None
        if (masked_tokens is not None) and positioned:
            masked_lm_positions_dim = masked_lm_positions.dim()
            if masked_lm_positions_dim == 2:
                position_ids = masked_lm_positions
                sequence_output = gather_positions(sequence_output, masked_lm_positions)
                # Well, `ignore_index` may vary with this case
                masked_tokens = masked_lm_ids
            elif masked_lm_positions_dim == 1:
                position_ids = position_ids.view(-1)[masked_lm_positions]
                sequence_output = sequence_output.contiguous().view(-1, sequence_output.size(-1))[masked_lm_positions]
                masked_tokens = masked_lm_ids
            else:
                raise Exception("Invalid dim of masked_lm_positions and masked_lm_ids")

        pred_score, seq_score = self.cls(sequence_output, pooled_output)

        loss = 0.0
        mlm_logits = pred_score.view(-1, self.vocab_size)
        if masked_tokens is not None:
            mlm_labels = masked_tokens.view(-1)
            loss = self.loss_function(mlm_logits, mlm_labels)
            if not self._omit_other_output:
                self._update_local_metrics(mlm_logits, mlm_labels)
                self.local_metrics["local_mlm_loss"] = loss.item()
        if sentence_label is not None:
            next_sentence_loss = self.nsp_loss_function(seq_score.view(-1, 2), sentence_label.view(-1))
            if not self._omit_other_output:
                self.local_metrics["local_nsp_loss"] = next_sentence_loss.item()
            loss = loss + next_sentence_loss
        if self._use_moe:
            for i in range(len(self.encoder.l_aux)):
                loss += self.extra_da_transformer_config.get("moe_l_aux_factor", 0.01) * self.encoder.l_aux[i]
                loss += self.extra_da_transformer_config.get("moe_z_loss_factor", 0.0) * self.encoder.z_loss[i]
                self.local_metrics["l_aux" + str(i)] = self.encoder.l_aux[i].item()
                self.local_metrics["z_loss" + str(i)] = self.encoder.z_loss[i].item()
        if self._use_moe_attn:
            for i in range(len(self.encoder.l_aux_attn)):
                loss += (
                    self.extra_da_transformer_config.get("moe_l_aux_factor_attn", 0.01) * self.encoder.l_aux_attn[i]
                )

        # print('encoder_last_hidden_state', encoder_last_seq_output)
        # print('decoder_last_hidden_state', decoder_last_seq_output)
        if self._omit_other_output:
            return {"loss": loss}
        return {
            "loss": loss,
            "encoder_last_hidden_state": encoder_last_seq_output,
            "decoder_last_hidden_state": decoder_last_seq_output,
            "pooled_output": pooled_output,
        }


class DebertaModelPinyin(DebertaBarePinyin):
    def __init__(
        self,
        max_len: int = 512,
        abs_pos_embedding: bool = False,
        ignore_index: int = -1,
        calc_mlm_accuracy: bool = True,
        tie_embedding: bool = True,
        use_emd: bool = False,
        num_emd_groups: int = 1,
        emd_group_repeat: int = 2,
        layernorm_fp16: bool = False,
        use_fast: bool = False,
        head_layernorm_type: str = "default",
        omit_other_output: bool = False,
        **option,
    ):
        # vocab_size and pinyin_vocab_size must be in **option
        super().__init__(
            use_fast=use_fast,
            layernorm_fp16=layernorm_fp16,
            omit_other_output=omit_other_output,
            **option,
        )

        if abs_pos_embedding:
            self.ape = nn.Embedding(max_len, option["dim"])
        else:
            self.ape = None
        self.cls = BertPreTrainingHeads(
            dict(
                dim=option["dim"],
                embedding_dim=option["embedding_dim"],
                layer_norm_eps=option["layer_norm_eps"],
                vocab_size=option["vocab_size"],
                act=option["act"],
                layernorm_type=head_layernorm_type,
            )
        )
        if layernorm_fp16:
            self.cls.predictions.layer_norm._simply_cast = True
        if tie_embedding:
            self._tie_weights()

        self._omit_other_output = omit_other_output
        self._calc_mlm_accuracy = calc_mlm_accuracy and not self._omit_other_output
        self._use_moe = self.encoder.config.use_moe
        self._use_moe_attn = self.encoder.config.use_moe_attn
        self.mlm_accuracy = self._calc_mlm_accuracy and CategoricalAccuracy()

        self.ignore_index = ignore_index
        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.nsp_loss_function = torch.nn.CrossEntropyLoss()

        self.local_metrics = {}

        self.use_emd = use_emd
        if use_emd:
            self.emd = DebertaEMD(
                self.da_config,
                num_emd_groups=num_emd_groups,
                emd_group_repeat=emd_group_repeat,
                use_fast=use_fast,
            )

        self.apply(partial(init_weights, initializer_range=self.initializer_range))

    def _tie_weights(self):
        self.cls.predictions.decoder.weight = self.embedding.token_embedder_tokens.weight

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        global_metrics = {}
        if self._calc_mlm_accuracy:
            global_metrics["mlm_accuracy"] = self.mlm_accuracy.get_metric(reset)
        global_metrics.update(self.local_metrics)
        return global_metrics

    def _update_local_metrics(self, mlm_logits, mlm_labels):
        if self._calc_mlm_accuracy:
            total_count, correct_count = float(self.mlm_accuracy.total_count), float(self.mlm_accuracy.correct_count)
            mlm_positions = torch.nonzero(mlm_labels != self.ignore_index, as_tuple=False).view(-1)
            self.mlm_accuracy(mlm_logits[mlm_positions], mlm_labels[mlm_positions])
            local_total_count = float(self.mlm_accuracy.total_count) - total_count
            local_correct_count = float(self.mlm_accuracy.correct_count) - correct_count
            local_accuracy = 0.0 if local_total_count == 0 else (float(local_correct_count) / float(local_total_count))
            self.local_metrics.update(
                {
                    "local_mlm_total_count": local_total_count,
                    "local_mlm_correct_count": local_correct_count,
                    "local_mlm_accuracy": local_accuracy,
                }
            )

    def forward(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
        masked_tokens=None,
        sentence_label=None,
        masked_lm_positions=None,
        masked_lm_ids=None,
        output_pooled=False,
        pinyin_ids=None,
    ):
        if attention_mask is not None:
            mask = attention_mask
        else:
            mask = input_ids != self.padding_index
            mask[:, 0:1] = 1

        output = super().forward(
            input_ids=input_ids,
            attention_mask=mask,
            token_type_ids=token_type_ids,
            # position_ids=position_ids,
            output_pooled=output_pooled or sentence_label is not None,
            output_rel_pos=self.use_emd,
            pinyin_ids=pinyin_ids,
        )

        sequence_output = output["sequence_output"]
        pooled_output = output["pooled_output"]

        encoder_last_seq_output = sequence_output

        if self.use_emd:
            if self.ape is not None and position_ids is not None:
                abs_pos_embeddings = self.ape(position_ids.long())
                sequence_output = self.emd(
                    abs_pos_embeddings + sequence_output,
                    sequence_output,
                    mask,
                    relative_pos=output["relative_pos"],
                    rel_embedding=self.encoder.rel_embeddings.weight,
                )
            else:
                # TODO: fix
                sequence_output = self.emd(sequence_output, sequence_output)

        decoder_last_seq_output = sequence_output
        # print('decoder_last_seq_output', sequence_output)

        if masked_tokens is None and sentence_label is None and masked_lm_positions is None and masked_lm_ids is None:
            return {
                "sequence_output": decoder_last_seq_output,
                "pooled_output": pooled_output,
            }

        # Shrink `sequence_output` and `masked_tokens` according to `masked_lm_positions` and `masked_lm_ids`
        positioned = masked_lm_positions is not None
        if (masked_tokens is not None) and positioned:
            masked_lm_positions_dim = masked_lm_positions.dim()
            if masked_lm_positions_dim == 2:
                position_ids = masked_lm_positions
                sequence_output = gather_positions(sequence_output, masked_lm_positions)
                # Well, `ignore_index` may vary with this case
                masked_tokens = masked_lm_ids
            elif masked_lm_positions_dim == 1:
                position_ids = position_ids.view(-1)[masked_lm_positions]
                sequence_output = sequence_output.contiguous().view(-1, sequence_output.size(-1))[masked_lm_positions]
                masked_tokens = masked_lm_ids
            else:
                raise Exception("Invalid dim of masked_lm_positions and masked_lm_ids")

        pred_score, seq_score = self.cls(sequence_output, pooled_output)

        loss = 0.0
        mlm_logits = pred_score.view(-1, self.vocab_size)
        if masked_tokens is not None:
            mlm_labels = masked_tokens.view(-1)
            loss = self.loss_function(mlm_logits, mlm_labels)
            if not self._omit_other_output:
                self._update_local_metrics(mlm_logits, mlm_labels)
                self.local_metrics["local_mlm_loss"] = loss.item()
        if sentence_label is not None:
            next_sentence_loss = self.nsp_loss_function(seq_score.view(-1, 2), sentence_label.view(-1))
            if not self._omit_other_output:
                self.local_metrics["local_nsp_loss"] = next_sentence_loss.item()
            loss = loss + next_sentence_loss
        if self._use_moe:
            for i in range(len(self.encoder.l_aux)):
                loss += self.extra_da_transformer_config.get("moe_l_aux_factor", 0.01) * self.encoder.l_aux[i]
                loss += self.extra_da_transformer_config.get("moe_z_loss_factor", 0.0) * self.encoder.z_loss[i]
                self.local_metrics["l_aux" + str(i)] = self.encoder.l_aux[i].item()
                self.local_metrics["z_loss" + str(i)] = self.encoder.z_loss[i].item()
        if self._use_moe_attn:
            for i in range(len(self.encoder.l_aux_attn)):
                loss += (
                    self.extra_da_transformer_config.get("moe_l_aux_factor_attn", 0.01) * self.encoder.l_aux_attn[i]
                )

        # print('encoder_last_hidden_state', encoder_last_seq_output)
        # print('decoder_last_hidden_state', decoder_last_seq_output)
        if self._omit_other_output:
            return {"loss": loss}
        return {
            "loss": loss,
            "encoder_last_hidden_state": encoder_last_seq_output,
            "decoder_last_hidden_state": decoder_last_seq_output,
            "pooled_output": pooled_output,
        }
