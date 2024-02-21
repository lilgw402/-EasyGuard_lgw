"""Reference to ptx bert implementations with slight changes: removing ft"""
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import embedding

from .activation import Activation
from .layernorm import LayerNorm, LayerNormTypes


def get_local_world_size() -> int:
    return int(os.getenv("LOCAL_WORLD_SIZE", "1"))


def get_local_node() -> int:
    rank = get_rank()
    return rank // get_local_world_size()


def create_process_group(**kwargs):
    local_node = get_local_node()
    num_nodes = get_world_size() // get_local_world_size()
    pgs = [torch.distributed.new_group(**kwargs) for _ in range(num_nodes)]
    return pgs[local_node]


def get_rank() -> int:
    return int(os.getenv("RANK", "0"))


def get_local_rank() -> int:
    return int(os.getenv("LOCAL_RANK", "0"))


def get_world_size() -> int:
    return int(os.getenv("WORLD_SIZE", "1"))


@torch.jit.script
def slice_pos_ids(position_ids: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    return position_ids[:, : input_ids.size(1)]


class Embedding(torch.nn.Module):
    """
    A more featureful embedding module than the default in Pytorch.  Adds the ability to:

        1. embed higher-order inputs
        2. pre-specify the weight matrix
        3. use a non-trainable embedding
        4. project the resultant embeddings to some other dimension (which only makes sense with
           non-trainable embeddings).
        5. build all of this easily ``from_option``

    Note that if you are using our data API and are trying to embed a TextField, you should use a
    TextFieldEmbedder instead of using this directly.

    Args:
        num_embeddings : int:
            Size of the dictionary of embeddings (vocabulary size).
        embedding_dim : int
            The size of each embedding vector.
        weight : torch.FloatTensor, (optional, default=None)
            A pre-initialised weight matrix for the embedding lookup, allowing the use of
            pretrained vectors.
        padding_index : int, (optional, default=None)
            If given, pads the output with zeros whenever it encounters the index.
        trainable : bool, (optional, default=True)
            Whether or not to optimize the embedding parameters.
        max_norm : float, (optional, default=None)
            If given, will renormalize the embeddings to always have a norm lesser than this
        norm_type : float, (optional, default=2):
            The p of the p-norm to compute for the max_norm option
        scale_grad_by_freq : boolean, (optional, default=False):
            If given, this will scale gradients by the frequency of the words in the mini-batch.
        sparse : bool, (optional, default=False):
            Whether or not the Pytorch backend should use a sparse representation of the embedding weight.

    Returns:
        An Embedding module.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        weight: torch.FloatTensor = None,
        padding_index: int = None,
        padding_idx: int = None,  # Compat with nn.Embedding
        trainable: bool = True,
        max_norm: float = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.padding_index = padding_index if padding_index is not None else padding_idx
        self.padding_idx = self.padding_index  # Compat with nn.Embedding
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        self.output_dim = embedding_dim

        if weight is None:
            weight = torch.FloatTensor(num_embeddings, embedding_dim)
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)
            torch.nn.init.xavier_uniform_(self.weight)
        else:
            if weight.size() != (num_embeddings, embedding_dim):
                raise Exception("A weight matrix was passed with contradictory embedding shapes.")
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)

        if self.padding_index is not None:
            self.weight.data[self.padding_index].fill_(0)

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return embedding(
            tokens,
            self.weight,
            padding_idx=self.padding_index,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )


class BertEmbedding(torch.nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(
        self,
        dim: int,
        vocab_size: int,
        n_segments: int,
        max_len: int,
        p_drop_hidden: float = 0.1,
        padding_index: int = None,
        layer_norm_eps: float = 1e-12,
        token_embedding_dim: int = None,
        output_dim: int = None,
        adaptive_option: Optional[dict] = None,
        layernorm_type: str = "v0",
        position_offset: int = 0,
        use_dist_token_update: bool = False,
        max_seq_len: int = 512,
        dist_token_update_option: Optional[Dict] = None,
    ):
        super().__init__()

        token_embedding_dim = token_embedding_dim or dim

        dist_token_update_option = dist_token_update_option or {}
        dist_token_method = dist_token_update_option.get("method", "v0")
        use_sparse_embedding = dist_token_update_option.get("use_sparse_embedding")
        if use_sparse_embedding is None:
            if not use_dist_token_update:
                use_sparse_embedding = False
            else:
                use_sparse_embedding = dist_token_method != "v0"

        if token_embedding_dim != dim:
            assert not adaptive_option, "Cannot use `adaptive_option` when `token_embedding_dim` is set."
            self.token_embedder_tokens = Embedding(vocab_size, token_embedding_dim, padding_index=padding_index)
            self.token_embedding_proj = nn.Linear(token_embedding_dim, dim)
        else:
            self.token_embedding_proj = None
            if not adaptive_option:
                self.token_embedder_tokens = Embedding(
                    vocab_size,
                    dim,
                    padding_index=padding_index,
                    sparse=use_sparse_embedding,
                )
            else:
                self.token_embedder_tokens = AdaptiveEmbedding(
                    vocab_size,
                    dim,
                    padding_idx=padding_index,
                    **adaptive_option,
                )

        self.position_offset = position_offset or 0
        self.token_embedder_positions = Embedding(max_len + self.position_offset, dim) if max_len else None
        self.token_embedder_segments = Embedding(n_segments, dim) if n_segments else None

        self.dim = dim
        self.output_dim = output_dim or dim
        if self.output_dim != self.dim:
            self.out_proj = nn.Linear(self.dim, self.output_dim, bias=False)
        else:
            self.out_proj = None

        self.norm = (LayerNormTypes[layernorm_type])(self.output_dim, eps=layer_norm_eps)
        self.dropout = torch.nn.Dropout(p_drop_hidden)

        if max_len:
            # position_ids (1, len position emb) is contiguous in memory and exported when serialized
            self.register_buffer(
                "position_ids",
                torch.arange(max_len + self.position_offset).expand((1, -1)),
            )

        self.add_pos_embedding = True
        self.add_seg_embedding = True

        dist_token_absent_grad_zero = dist_token_update_option.get("set_absent_grad_zero", True)
        dist_token_clip_only = dist_token_update_option.get("use_clip_only", False)
        dist_token_grad_by_count = dist_token_update_option.get("avg_grad_by_count", True)
        dist_token_grad_fp32 = dist_token_update_option.get("use_grad_fp32", True)
        self._use_dist_token_update = use_dist_token_update
        self._max_seq_len = max_seq_len
        self._cached_input_ids = None
        self._use_cached_input_ids = self._use_dist_token_update and (dist_token_method == "v0")
        if self._use_dist_token_update:
            world_size = get_world_size()
            cur_rank = get_rank()
            pg = None
            if dist_token_method == "v1":
                pg = create_process_group(backend="gloo")

            def all_gather_embedding_grad_v0(grad):
                if getattr(self, "_cached_input_ids", None) is None:
                    return grad
                emp_grad = torch.zeros_like(grad)
                if dist_token_absent_grad_zero or dist_token_clip_only:
                    cur_ids = self._cached_input_ids.unique()
                    emp_grad[cur_ids] = grad[cur_ids]
                    grad = emp_grad
                    if dist_token_clip_only:
                        self._cached_input_ids = None
                        return grad
                cur_ids = F.pad(
                    self._cached_input_ids,
                    (
                        0,
                        self._max_seq_len - self._cached_input_ids.size(1),
                        0,
                        0,
                    ),
                    "constant",
                    -1,
                )
                # print(f'local input_ids: {self._cached_input_ids.shape}')
                self._cached_input_ids = None
                all_ids = [torch.full_like(cur_ids, -1) for _ in range(world_size)]
                torch.distributed.all_gather(all_ids, cur_ids)  # use `all_gather_object` instead?
                avg_grad_by = world_size
                if not dist_token_grad_by_count:
                    uni_ids = torch.cat(all_ids).unique()  # dedup, sort
                    if uni_ids[0].item() == -1:
                        uni_ids = uni_ids[1:]  # remove `-1`
                else:
                    uni_ids, uni_counts = torch.cat(all_ids).unique(return_counts=True)  # dedup, sort
                    if uni_ids[0].item() == -1:
                        uni_ids = uni_ids[1:]  # remove `-1`
                        uni_counts = uni_counts[1:]
                    avg_grad_by = uni_counts.unsqueeze(1)
                cur_grad = grad[uni_ids]
                cur_grad[cur_grad != cur_grad] = 0
                # print(f'before allreduce: {uni_ids.shape}, {cur_grad.max()}, {cur_grad.min()}')
                if dist_token_grad_fp32:
                    cur_grad = cur_grad.float()
                torch.distributed.all_reduce(cur_grad)
                # print(f'after allreduce: {uni_ids.shape}, {cur_grad.max()}, {cur_grad.min()}')
                all_grad = cur_grad / avg_grad_by
                if dist_token_grad_fp32:
                    emp_grad[uni_ids] = all_grad.to(dtype=emp_grad.dtype)
                else:
                    emp_grad[uni_ids] = all_grad
                # TODO: check nan/inf ?
                # emp_grad[emp_grad != emp_grad] = 0
                return emp_grad

            def all_gather_embedding_grad_v1(grad):
                # print(f'grad before, {grad.is_coalesced()}')
                # print(f'grad index: {grad._indices()}')
                # print(f'grad value: {grad._values()}')
                if dist_token_grad_fp32:
                    grad = grad.float()
                torch.distributed.all_reduce(grad, group=pg)
                grad = grad / world_size
                if dist_token_grad_fp32:
                    grad = grad.half()
                grad = grad.coalesce()
                # grad = grad.to_dense()  # Cannot, because grad tensor type must not change
                # print(grad)
                return grad

            def all_gather_embedding_grad_v2(grad):
                # print(f'grad before, {grad.is_coalesced()}')
                grad = grad.coalesce()
                all_grads = [None for _ in range(world_size)]
                torch.distributed.all_gather_object(all_grads, grad)
                a_grad = None
                for i_grad in all_grads:
                    if not dist_token_grad_fp32:
                        i_grad = i_grad.to(device=grad.device)
                    else:
                        i_grad = i_grad.to(device=grad.device, dtype=torch.float32)
                    # print(f'grad after, {i_grad.is_coalesced()}')
                    # i_grad = i_grad.coalesce()
                    # i_indices = i_grad.indices()
                    # i_values = i_grad.values()
                    if a_grad is None:
                        a_grad = i_grad
                    else:
                        a_grad.add_(i_grad)
                grad = a_grad / world_size
                if dist_token_grad_fp32:
                    grad = grad.half()
                grad = grad.coalesce()
                # print(grad)
                return grad

            def all_gather_embedding_grad_v3(grad):
                # print(f'grad before, {grad.is_coalesced()}')
                grad = grad.coalesce()
                cur_size = grad.size()
                cur_indices = grad.indices()
                cur_indices = cur_indices.squeeze(0)
                cur_values = grad.values()

                cur_seqlen = cur_indices.size(0)
                all_seqlen = [torch.tensor([0], device=grad.device) for _ in range(world_size)]
                torch.distributed.all_gather(all_seqlen, torch.tensor([cur_seqlen], device=grad.device))

                max_seqlen = torch.cat(all_seqlen).max().item()
                if max_seqlen > cur_seqlen:
                    cur_idx = torch.cat(
                        (
                            cur_indices,
                            torch.full(
                                (max_seqlen - cur_seqlen,),
                                -1,
                                dtype=cur_indices.dtype,
                                device=cur_indices.device,
                            ),
                        )
                    )
                    cur_val = torch.cat(
                        (
                            cur_values,
                            torch.zeros(
                                (max_seqlen - cur_seqlen, cur_values.size(1)),
                                dtype=cur_values.dtype,
                                device=cur_values.device,
                            ),
                        )
                    )
                else:
                    cur_idx = cur_indices
                    cur_val = cur_values

                all_ids = [torch.full_like(cur_idx, -1) for _ in range(world_size)]
                torch.distributed.all_gather(all_ids, cur_idx)

                all_grads = [torch.zeros_like(cur_val) for _ in range(world_size)]
                torch.distributed.all_gather(all_grads, cur_val)

                for i, (idx, val) in enumerate(zip(all_ids, all_grads)):
                    if i == cur_rank:
                        continue
                    this_seqlen = all_seqlen[i].item()
                    this_grad = torch.sparse_coo_tensor(
                        idx[:this_seqlen].unsqueeze(0),
                        val[:this_seqlen],
                        size=cur_size,
                        dtype=grad.dtype,
                        device=grad.device,
                    )
                    grad.add_(this_grad)

                grad = grad / world_size
                grad = grad.coalesce()
                # print(grad)
                return grad

            dist_token_methods = {
                "v0": all_gather_embedding_grad_v0,
                "v1": all_gather_embedding_grad_v1,
                "v2": all_gather_embedding_grad_v2,
                "v3": all_gather_embedding_grad_v3,
            }

            self.token_embedder_tokens.weight.register_hook(dist_token_methods[dist_token_method])

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        mask=None,
        embed_only: bool = False,
    ):
        if isinstance(input_ids, dict):
            token_type_ids = input_ids.get("segments")
            position_ids = input_ids.get("positions")
            inputs_embeds = input_ids.get("token_embeddings")
            input_ids = input_ids.get("tokens")

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if self.training and self._use_cached_input_ids:
            self._cached_input_ids = input_ids

        if inputs_embeds is None:
            embeddings = self.token_embedder_tokens(input_ids)
            if self.token_embedding_proj is not None:
                embeddings = self.token_embedding_proj(embeddings)
        else:
            embeddings = inputs_embeds

        if (self.token_embedder_positions is not None) and self.add_pos_embedding:
            if position_ids is None:
                # position_ids = self.position_ids[:, :input_ids.size(1)]
                position_ids = slice_pos_ids(self.position_ids, input_ids)
            if self.position_offset:
                position_ids = position_ids + self.position_offset
            position_embeddings = self.token_embedder_positions(position_ids)
            embeddings += position_embeddings

        if (self.token_embedder_segments is not None) and self.add_seg_embedding:
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=embeddings.device)
            token_type_embeddings = self.token_embedder_segments(token_type_ids)
            embeddings += token_type_embeddings

        if embed_only:
            return embeddings

        if self.out_proj is not None:
            embeddings = self.out_proj(embeddings)

        embeddings = self.norm(embeddings)

        if mask is not None:
            if mask.dim() != embeddings.dim():
                if mask.dim() == 4:
                    mask = mask.squeeze(1).squeeze(1)
                mask = mask.unsqueeze(2)
            embeddings *= mask.to(embeddings.dtype)

        embeddings = self.dropout(embeddings)
        return embeddings


"""Refer to chineseBERT implementations with slight changes """


class PinyinEmbedding(nn.Module):
    def __init__(self, embedding_size: int, pinyin_out_dim: int, pinyin_vocab_size: int):
        """
            Pinyin Embedding Module
        Args:
            embedding_size: the size of each embedding vector
            pinyin_out_dim: kernel number of conv
        """
        super(PinyinEmbedding, self).__init__()
        self.pinyin_out_dim = pinyin_out_dim
        self.embedding = nn.Embedding(pinyin_vocab_size, embedding_size)
        self.conv1 = nn.Conv1d(
            in_channels=embedding_size,
            out_channels=self.pinyin_out_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, pinyin_ids):
        """
        Args:
            pinyin_ids: (bs*sentence_length*pinyin_locs)

        Returns:
            pinyin_embed: (bs,sentence_length,pinyin_out_dim)
        """
        # input pinyin ids for 1-D conv
        embed = self.embedding(pinyin_ids)  # [bs,sentence_length,pinyin_locs,embed_size]
        bs, sentence_length, pinyin_locs, embed_size = embed.shape
        view_embed = embed.view(-1, pinyin_locs, embed_size)  # [(bs*sentence_length),pinyin_locs,embed_size]
        input_embed = view_embed.permute(0, 2, 1)  # [(bs*sentence_length), embed_size, pinyin_locs]
        # conv + max_pooling
        pinyin_conv1 = self.conv1(input_embed)  # [(bs*sentence_length),pinyin_out_dim,H]
        pinyin_embed1 = F.max_pool1d(pinyin_conv1, pinyin_conv1.shape[-1])  # [(bs*sentence_length),pinyin_out_dim,1]
        view_pinyin_embed = pinyin_embed1.view(
            bs, sentence_length, self.pinyin_out_dim
        )  # [bs,sentence_length,pinyin_out_dim]
        return view_pinyin_embed


class PinyinEmbeddingV2(nn.Module):
    def __init__(self, embedding_size: int, pinyin_out_dim: int, pinyin_vocab_size: int):
        """
            Pinyin Embedding Module
        Args:
            embedding_size: the size of each embedding vector
            pinyin_out_dim: kernel number of conv
        """
        super(PinyinEmbeddingV2, self).__init__()
        self.pinyin_out_dim = pinyin_out_dim
        self.embedding = nn.Embedding(pinyin_vocab_size, embedding_size)
        self.conv1 = nn.Conv1d(
            in_channels=embedding_size,
            out_channels=self.pinyin_out_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv2 = nn.Conv1d(
            in_channels=embedding_size,
            out_channels=self.pinyin_out_dim,
            kernel_size=2,
            stride=1,
            padding=0,
        )
        self.fc = nn.Linear(in_features=pinyin_out_dim * 2, out_features=pinyin_out_dim)

    def forward(self, pinyin_ids):
        """
        Args:
            pinyin_ids: (bs*sentence_length*pinyin_locs)

        Returns:
            pinyin_embed: (bs,sentence_length,pinyin_out_dim)
        """
        # input pinyin ids for 1-D conv
        embed = self.embedding(pinyin_ids)  # [bs,sentence_length,pinyin_locs,embed_size]
        bs, sentence_length, pinyin_locs, embed_size = embed.shape
        view_embed = embed.view(-1, pinyin_locs, embed_size)  # [(bs*sentence_length),pinyin_locs,embed_size]
        input_embed = view_embed.permute(0, 2, 1)  # [(bs*sentence_length), embed_size, pinyin_locs]
        # conv + max_pooling
        pinyin_conv1 = self.conv1(input_embed)  # [(bs*sentence_length),pinyin_out_dim,H]
        pinyin_embed1 = F.max_pool1d(pinyin_conv1, pinyin_conv1.shape[-1])  # [(bs*sentence_length),pinyin_out_dim,1]
        pinyin_conv2 = self.conv2(input_embed)  # [(bs*sentence_length),pinyin_out_dim,H]
        pinyin_embed2 = F.max_pool1d(pinyin_conv2, pinyin_conv2.shape[-1])  # [(bs*sentence_length),pinyin_out_dim,1]
        pinyin_embed = torch.cat((pinyin_embed1, pinyin_embed2), dim=1)  # [(bs*sentence_length),pinyin_out_dim*2,1]
        view_pinyin_embed = pinyin_embed.view(
            bs, sentence_length, self.pinyin_out_dim * 2
        )  # [bs,sentence_length,pinyin_out_dim*2]
        view_pinyin_embed = self.fc(view_pinyin_embed)  # [bs,sentence_length,pinyin_out_dim]
        return view_pinyin_embed


class BertEmbeddingPinyin(torch.nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(
        self,
        dim: int,
        vocab_size: int,
        pinyin_vocab_size: int,
        n_segments: int,
        max_len: int,
        p_drop_hidden: float = 0.1,
        padding_index: int = None,
        layer_norm_eps: float = 1e-12,
        token_embedding_dim: int = None,
        output_dim: int = None,
        adaptive_option: Optional[dict] = None,
        layernorm_type: str = "v0",
        position_offset: int = 0,
        use_dist_token_update: bool = False,
        max_seq_len: int = 512,
        dist_token_update_option: Optional[Dict] = None,
        pinyin_embedding_dim=128,
    ):
        super().__init__()

        token_embedding_dim = token_embedding_dim or dim

        dist_token_update_option = dist_token_update_option or {}
        dist_token_method = dist_token_update_option.get("method", "v0")
        use_sparse_embedding = dist_token_update_option.get("use_sparse_embedding")
        if use_sparse_embedding is None:
            if not use_dist_token_update:
                use_sparse_embedding = False
            else:
                use_sparse_embedding = dist_token_method != "v0"

        if token_embedding_dim != dim:
            assert not adaptive_option, "Cannot use `adaptive_option` when `token_embedding_dim` is set."
            self.token_embedder_tokens = Embedding(vocab_size, token_embedding_dim, padding_index=padding_index)
            self.token_embedding_proj = nn.Linear(token_embedding_dim, dim)
        else:
            self.token_embedding_proj = None
            if not adaptive_option:
                self.token_embedder_tokens = Embedding(
                    vocab_size,
                    dim,
                    padding_index=padding_index,
                    sparse=use_sparse_embedding,
                )
            else:
                self.token_embedder_tokens = AdaptiveEmbedding(
                    vocab_size,
                    dim,
                    padding_idx=padding_index,
                    **adaptive_option,
                )

        self.position_offset = position_offset or 0
        self.token_embedder_positions = Embedding(max_len + self.position_offset, dim) if max_len else None
        self.token_embedder_segments = Embedding(n_segments, dim) if n_segments else None

        self.dim = dim
        self.output_dim = output_dim or dim
        if self.output_dim != self.dim:
            self.out_proj = nn.Linear(self.dim, self.output_dim, bias=False)
        else:
            self.out_proj = None

        self.norm = (LayerNormTypes[layernorm_type])(self.output_dim, eps=layer_norm_eps)
        self.dropout = torch.nn.Dropout(p_drop_hidden)

        if max_len:
            # position_ids (1, len position emb) is contiguous in memory and exported when serialized
            self.register_buffer(
                "position_ids",
                torch.arange(max_len + self.position_offset).expand((1, -1)),
            )

        self.add_pos_embedding = True
        self.add_seg_embedding = True

        dist_token_absent_grad_zero = dist_token_update_option.get("set_absent_grad_zero", True)
        dist_token_clip_only = dist_token_update_option.get("use_clip_only", False)
        dist_token_grad_by_count = dist_token_update_option.get("avg_grad_by_count", True)
        dist_token_grad_fp32 = dist_token_update_option.get("use_grad_fp32", True)
        self._use_dist_token_update = use_dist_token_update
        self._max_seq_len = max_seq_len
        self._cached_input_ids = None
        self._use_cached_input_ids = self._use_dist_token_update and (dist_token_method == "v0")

        # pinyin Embeddings
        self.pinyin_embeddings = PinyinEmbeddingV2(
            embedding_size=pinyin_embedding_dim,
            pinyin_out_dim=dim,
            pinyin_vocab_size=pinyin_vocab_size,
        )
        # for concat[token_emb, pinyin_emb] projection
        self.map_fc = nn.Linear(dim * 2, dim)

        if self._use_dist_token_update:
            world_size = get_world_size()
            cur_rank = get_rank()
            pg = None
            if dist_token_method == "v1":
                pg = create_process_group(backend="gloo")

            def all_gather_embedding_grad_v0(grad):
                if getattr(self, "_cached_input_ids", None) is None:
                    return grad
                emp_grad = torch.zeros_like(grad)
                if dist_token_absent_grad_zero or dist_token_clip_only:
                    cur_ids = self._cached_input_ids.unique()
                    emp_grad[cur_ids] = grad[cur_ids]
                    grad = emp_grad
                    if dist_token_clip_only:
                        self._cached_input_ids = None
                        return grad
                cur_ids = F.pad(
                    self._cached_input_ids,
                    (
                        0,
                        self._max_seq_len - self._cached_input_ids.size(1),
                        0,
                        0,
                    ),
                    "constant",
                    -1,
                )
                # print(f'local input_ids: {self._cached_input_ids.shape}')
                self._cached_input_ids = None
                all_ids = [torch.full_like(cur_ids, -1) for _ in range(world_size)]
                torch.distributed.all_gather(all_ids, cur_ids)  # use `all_gather_object` instead?
                avg_grad_by = world_size
                if not dist_token_grad_by_count:
                    uni_ids = torch.cat(all_ids).unique()  # dedup, sort
                    if uni_ids[0].item() == -1:
                        uni_ids = uni_ids[1:]  # remove `-1`
                else:
                    uni_ids, uni_counts = torch.cat(all_ids).unique(return_counts=True)  # dedup, sort
                    if uni_ids[0].item() == -1:
                        uni_ids = uni_ids[1:]  # remove `-1`
                        uni_counts = uni_counts[1:]
                    avg_grad_by = uni_counts.unsqueeze(1)
                cur_grad = grad[uni_ids]
                cur_grad[cur_grad != cur_grad] = 0
                # print(f'before allreduce: {uni_ids.shape}, {cur_grad.max()}, {cur_grad.min()}')
                if dist_token_grad_fp32:
                    cur_grad = cur_grad.float()
                torch.distributed.all_reduce(cur_grad)
                # print(f'after allreduce: {uni_ids.shape}, {cur_grad.max()}, {cur_grad.min()}')
                all_grad = cur_grad / avg_grad_by
                if dist_token_grad_fp32:
                    emp_grad[uni_ids] = all_grad.to(dtype=emp_grad.dtype)
                else:
                    emp_grad[uni_ids] = all_grad
                # TODO: check nan/inf ?
                # emp_grad[emp_grad != emp_grad] = 0
                return emp_grad

            def all_gather_embedding_grad_v1(grad):
                # print(f'grad before, {grad.is_coalesced()}')
                # print(f'grad index: {grad._indices()}')
                # print(f'grad value: {grad._values()}')
                if dist_token_grad_fp32:
                    grad = grad.float()
                torch.distributed.all_reduce(grad, group=pg)
                grad = grad / world_size
                if dist_token_grad_fp32:
                    grad = grad.half()
                grad = grad.coalesce()
                # grad = grad.to_dense()  # Cannot, because grad tensor type must not change
                # print(grad)
                return grad

            def all_gather_embedding_grad_v2(grad):
                # print(f'grad before, {grad.is_coalesced()}')
                grad = grad.coalesce()
                all_grads = [None for _ in range(world_size)]
                torch.distributed.all_gather_object(all_grads, grad)
                a_grad = None
                for i_grad in all_grads:
                    if not dist_token_grad_fp32:
                        i_grad = i_grad.to(device=grad.device)
                    else:
                        i_grad = i_grad.to(device=grad.device, dtype=torch.float32)
                    # print(f'grad after, {i_grad.is_coalesced()}')
                    # i_grad = i_grad.coalesce()
                    # i_indices = i_grad.indices()
                    # i_values = i_grad.values()
                    if a_grad is None:
                        a_grad = i_grad
                    else:
                        a_grad.add_(i_grad)
                grad = a_grad / world_size
                if dist_token_grad_fp32:
                    grad = grad.half()
                grad = grad.coalesce()
                # print(grad)
                return grad

            def all_gather_embedding_grad_v3(grad):
                # print(f'grad before, {grad.is_coalesced()}')
                grad = grad.coalesce()
                cur_size = grad.size()
                cur_indices = grad.indices()
                cur_indices = cur_indices.squeeze(0)
                cur_values = grad.values()

                cur_seqlen = cur_indices.size(0)
                all_seqlen = [torch.tensor([0], device=grad.device) for _ in range(world_size)]
                torch.distributed.all_gather(all_seqlen, torch.tensor([cur_seqlen], device=grad.device))

                max_seqlen = torch.cat(all_seqlen).max().item()
                if max_seqlen > cur_seqlen:
                    cur_idx = torch.cat(
                        (
                            cur_indices,
                            torch.full(
                                (max_seqlen - cur_seqlen,),
                                -1,
                                dtype=cur_indices.dtype,
                                device=cur_indices.device,
                            ),
                        )
                    )
                    cur_val = torch.cat(
                        (
                            cur_values,
                            torch.zeros(
                                (max_seqlen - cur_seqlen, cur_values.size(1)),
                                dtype=cur_values.dtype,
                                device=cur_values.device,
                            ),
                        )
                    )
                else:
                    cur_idx = cur_indices
                    cur_val = cur_values

                all_ids = [torch.full_like(cur_idx, -1) for _ in range(world_size)]
                torch.distributed.all_gather(all_ids, cur_idx)

                all_grads = [torch.zeros_like(cur_val) for _ in range(world_size)]
                torch.distributed.all_gather(all_grads, cur_val)

                for i, (idx, val) in enumerate(zip(all_ids, all_grads)):
                    if i == cur_rank:
                        continue
                    this_seqlen = all_seqlen[i].item()
                    this_grad = torch.sparse_coo_tensor(
                        idx[:this_seqlen].unsqueeze(0),
                        val[:this_seqlen],
                        size=cur_size,
                        dtype=grad.dtype,
                        device=grad.device,
                    )
                    grad.add_(this_grad)

                grad = grad / world_size
                grad = grad.coalesce()
                # print(grad)
                return grad

            dist_token_methods = {
                "v0": all_gather_embedding_grad_v0,
                "v1": all_gather_embedding_grad_v1,
                "v2": all_gather_embedding_grad_v2,
                "v3": all_gather_embedding_grad_v3,
            }

            self.token_embedder_tokens.weight.register_hook(dist_token_methods[dist_token_method])

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        mask=None,
        embed_only: bool = False,
        pinyin_ids=None,
    ):
        if isinstance(input_ids, dict):
            token_type_ids = input_ids.get("segments")
            position_ids = input_ids.get("positions")
            inputs_embeds = input_ids.get("token_embeddings")
            input_ids = input_ids.get("tokens")

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if self.training and self._use_cached_input_ids:
            self._cached_input_ids = input_ids

        if inputs_embeds is None:
            embeddings = self.token_embedder_tokens(input_ids)
            if self.token_embedding_proj is not None:
                embeddings = self.token_embedding_proj(embeddings)
        else:
            embeddings = inputs_embeds
        pinyin_embeddings = self.pinyin_embeddings(pinyin_ids)
        concat_token_embeddings = torch.cat((embeddings, pinyin_embeddings), 2)
        embeddings = self.map_fc(concat_token_embeddings)

        if (self.token_embedder_positions is not None) and self.add_pos_embedding:
            if position_ids is None:
                # position_ids = self.position_ids[:, :input_ids.size(1)]
                position_ids = slice_pos_ids(self.position_ids, input_ids)
            if self.position_offset:
                position_ids = position_ids + self.position_offset
            position_embeddings = self.token_embedder_positions(position_ids)
            embeddings += position_embeddings

        if (self.token_embedder_segments is not None) and self.add_seg_embedding:
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=embeddings.device)
            token_type_embeddings = self.token_embedder_segments(token_type_ids)
            embeddings += token_type_embeddings

        if embed_only:
            return embeddings

        if self.out_proj is not None:
            embeddings = self.out_proj(embeddings)

        embeddings = self.norm(embeddings)

        if mask is not None:
            if mask.dim() != embeddings.dim():
                if mask.dim() == 4:
                    mask = mask.squeeze(1).squeeze(1)
                mask = mask.unsqueeze(2)
            embeddings *= mask.to(embeddings.dtype)

        embeddings = self.dropout(embeddings)
        return embeddings


class AdaptiveEmbedding(nn.Module):
    """
    https://github.com/taufique74/AdaptiveIO/blob/0fd05695fd599b2706934959b76bb5ccb55521f3/model.py#L209
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        cutoffs: Optional[List[int]] = None,
        div_value: float = 2.0,
        head_bias: bool = False,
        tail_drop: float = 0.5,
    ):
        super().__init__()
        if not cutoffs:
            cutoffs = [5000, 10000]
        cutoffs = list(cutoffs)

        if (
            (cutoffs != sorted(cutoffs))
            or (min(cutoffs) <= 0)
            or (max(cutoffs) >= (num_embeddings - 1))
            or (len(set(cutoffs)) != len(cutoffs))
            or any([int(c) != c for c in cutoffs])
        ):
            raise ValueError(
                "cutoffs should be a sequence of unique, positive "
                "integers sorted in an increasing order, where "
                "each value is between 1 and num_embeddings-1"
            )

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.cutoffs = cutoffs + [num_embeddings]
        self.div_value = div_value
        self.head_bias = head_bias
        self.tail_drop = tail_drop

        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.cutoffs[0]

        self.head = nn.Embedding(self.head_size, self.embedding_dim, padding_idx=padding_idx)

        self.tail = nn.ModuleList()

        for i in range(self.n_clusters):
            hsz = int(self.embedding_dim // (self.div_value ** (i + 1)))
            osz = self.cutoffs[i + 1] - self.cutoffs[i]

            projection = nn.Sequential(
                nn.Embedding(osz, hsz, padding_idx=padding_idx),
                nn.Linear(hsz, self.embedding_dim, bias=False),
                nn.Dropout(self.tail_drop),
            )

            self.tail.append(projection)

    def forward(self, input):
        used_rows = 0
        input_size = list(input.size())

        output = input.new_zeros([input.size(0) * input.size(1)] + [self.embedding_dim]).to(
            dtype=self.head.weight.dtype
        )
        input = input.view(-1)

        cutoff_values = [0] + self.cutoffs
        for i in range(len(cutoff_values) - 1):
            low_idx = cutoff_values[i]
            high_idx = cutoff_values[i + 1]

            input_mask = (input >= low_idx) & (input < high_idx)
            row_indices = input_mask.nonzero().squeeze()

            if row_indices.numel() == 0:
                continue
            out = self.head(input[input_mask] - low_idx) if i == 0 else self.tail[i - 1](input[input_mask] - low_idx)
            output.index_copy_(0, row_indices, out)
            used_rows += row_indices.numel()

        # if used_rows != input_size[0] * input_size[1]:
        #     raise RuntimeError("Target values should be in [0, {}], "
        #                        "but values in range [{}, {}] "
        #                        "were found. ".format(self.num_embeddings - 1,
        #                                              input.min().item(),
        #                                              input.max().item()))
        return output.view(input_size[0], input_size[1], -1)


def init_weights(module, initializer_range=0.02):
    if isinstance(module, torch.nn.Linear):
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, (torch.nn.Embedding, Embedding)):
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if getattr(module, "padding_idx", None) is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, LayerNorm):
        # ptx.ops.layernorm init itself
        pass
    elif isinstance(module, torch.nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class LMPredictionHead(torch.nn.Module):
    def __init__(self, option):
        """
        Args:
            option:
                dim:
                embedding_dim:
                layer_norm_eps:
                vocab_size:
                layernorm_type:
                act:
        """
        super().__init__()
        embedding_dim = option.get("embedding_dim") or option.get("dim")
        self.embedding_dim = embedding_dim
        self.dense = torch.nn.Linear(option.get("dim"), embedding_dim)
        self.activation = Activation(option.get("act", "gelu_new"))
        self.layer_norm = LayerNormTypes[option.get("layernorm_type", "v0")](embedding_dim)

        self.decoder = torch.nn.Linear(embedding_dim, option.get("vocab_size"), bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(option.get("vocab_size")))

    def forward(self, hidden_states):
        hidden_states = self.activation(self.dense(hidden_states))
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertPreTrainingHeads(torch.nn.Module):
    lm_prediction_head = LMPredictionHead

    def __init__(self, option):
        """
        Args:
            option:
                dim:
                embedding_dim:
                layer_norm_eps:
                vocab_size:
                layernorm_type:
        """
        super().__init__()
        self.predictions = self.lm_prediction_head(option)
        self.seq_relationship = torch.nn.Linear(option.get("dim"), 2)

    def forward(self, sequence_output, pooled_output=None):
        prediction_scores = self.predictions(sequence_output)  # MLM
        seq_relationship_score = None
        if pooled_output is not None:
            seq_relationship_score = self.seq_relationship(pooled_output)  # NSP
        return prediction_scores, seq_relationship_score


class BertPooler(torch.nn.Module):
    def __init__(self, option):
        """
        Args:
            option:
                dim:
        """
        super().__init__()
        self.dense = torch.nn.Linear(option.get("dim"), option.get("dim"))
        self.activation = torch.nn.Tanh()
        self.use_ft_linear = False

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        if self.use_ft_linear:
            first_token_tensor = first_token_tensor.contiguous()
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
