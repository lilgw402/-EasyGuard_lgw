# -*- coding: utf-8 -*-

"""
文本编码器代码文件；
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def _split_last(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    b, s, d = x.size()
    head_size = int(d / n_heads)
    return x.view(b, s, n_heads, head_size)


@torch.jit.script
def _merge_last(x: torch.Tensor) -> torch.Tensor:
    b, s, _, __ = x.size()
    return x.view(b, s, -1)


def gelu_new(x):
    """Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
    Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    # return 0.5 * x * (1 + torch.tanh(math.sqrt(math.pi / 2) * (x + 0.044715 * x ** 3)))


def tiny_value_of_dtype(dtype: torch.dtype):
    """
    Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
    issues such as division by zero.
    This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
    Only supports floating point dtypes.
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype in (torch.float, torch.double):
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))


class MultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""

    def __init__(self, config):
        """
        Args:
            prop:
                dim:
                p_drop_attn:
                n_heads:
        """
        super().__init__()
        self.proj_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.proj_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.proj_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.drop = nn.Dropout(config.hidden_dropout_prob)
        self.scores = None  # for visualization
        self.n_heads = config.num_attention_heads

    def forward(self, x, mask, kv=None):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q = self.proj_q(x)
        k = self.proj_k(x) if kv is None else self.proj_k(kv)
        v = self.proj_v(x) if kv is None else self.proj_v(kv)
        q = _split_last(q, self.n_heads).transpose(1, 2)
        k = _split_last(k, self.n_heads).transpose(1, 2)
        v = _split_last(v, self.n_heads).transpose(1, 2)
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / (int(k.size(-1)) ** 0.5)
        if mask is not None:
            mask = mask[:, None, None, :]
            scores -= 10000.0 * (1.0 - mask)
        scores = F.softmax(scores, dim=-1)
        self.scores = scores
        scores = self.drop(scores)
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = _merge_last(h)
        return h


class PositionWiseFeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""

    def __init__(self, config):
        """
        Args:
            prop:
                dim:
                dim_ff:
        """
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act = gelu_new
        # self.activ = lambda x: activ_fn(cfg.activ_fn, x)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(self.act(self.fc1(x)))  # Gelu


class AlbertLMPredictionHead(torch.nn.Module):
    """albert的预测head"""

    def __init__(self, config, embedding_weights):
        """
        Args:
            option:
                dim:
                embedding_dim:
                layer_norm_eps:
                vocab_size:
        """
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.embedding_size)
        self.activation = gelu_new
        self.layer_norm = nn.LayerNorm(config.embedding_size, eps=config.layernorm_eps)

        # self.decoder = torch.nn.Linear(config.embedding_size, config.vocab_size, bias=False)
        self.decoder = nn.Linear(embedding_weights.size(1), embedding_weights.size(0), bias=False)
        self.decoder.weight = embedding_weights

        self.bias = torch.nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.activation(self.dense(hidden_states))
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class Block(nn.Module):
    """Transformer Block"""

    def __init__(self, config):
        """
        Args:
            prop:
                is_decoder:
                dim:
                p_drop_hidden:
                p_drop_attn:
                n_heads:
                dim_ff:
        """
        super().__init__()
        self.attn = MultiHeadedSelfAttention(config)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layernorm_eps)
        self.pwff = PositionWiseFeedForward(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layernorm_eps)
        self.drop = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x, mask):
        # x = x.type_as(self.proj.weight)
        # mask = mask.type_as(self.proj.weight)
        x = x.to(dtype=self.proj.weight.dtype)
        mask = mask.to(dtype=self.proj.weight.dtype)
        h = self.attn(x, mask)
        h = self.norm1(x + self.drop(self.proj(h)))
        h = self.norm2(h + self.drop(self.pwff(h)))
        return h


class Transformer(nn.Module):
    """Transformer with Self-Attentive Blocks"""

    def __init__(self, config):
        """ """
        super().__init__()
        self.blocks = nn.ModuleList([self._create_layer(config) for _ in range(config.num_hidden_layers)])
        self.dim = config.hidden_size
        self.attn_weights = None

    def get_output_dim(self):
        return self.dim

    def _create_layer(self, option):
        return Block(option)
        # TODO: 下面的是built-in的，按沈科的意思是更快，不过参数命名要改，后面做
        # return TransformerEncoderLayer(
        #     option['dim'],
        #     option['n_heads'],
        #     dim_feedforward=option['dim_ff'],
        #     dropout=option.get('p_drop_hidden', 0.1),
        #     activation=option.get('act', 'gelu'),
        # )

    def forward(self, h, mask):
        all_layer_outputs = []
        all_attn_weights = []
        for block in self.blocks:
            h = block(h, mask)
            all_layer_outputs.append(h)
            all_attn_weights.append(block.attn.scores)
        self.attn_weights = all_attn_weights
        return all_layer_outputs, all_attn_weights


class BertPooler(nn.Module):
    """
    https://code.byted.org/nlp/ptx/blob/master/ptx/core/model.py#L110
    """

    def __init__(self, config):
        """
        Args:
            option:
                dim:
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertEmbedding(nn.Module):
    """
    https://code.byted.org/nlp/ptx/blob/master/ptx/core/bert.py#L340
    Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config, padding_index=2):
        super().__init__()

        self.project_embedding_first = config.project_embedding_first
        dim = config.hidden_size if self.project_embedding_first else config.embedding_size
        self.token_embedder_tokens = torch.nn.Embedding(
            config.vocab_size, config.embedding_size, padding_idx=padding_index
        )
        self.token_embedder_positions = torch.nn.Embedding(config.max_position_embeddings, dim)
        self.token_embedder_segments = torch.nn.Embedding(config.type_vocab_size, dim)

        self.norm = nn.LayerNorm(dim, eps=config.layernorm_eps)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        if config.embedding_size != config.hidden_size:
            self.proj_embedding_hidden = torch.nn.Linear(config.embedding_size, config.hidden_size)
        else:
            self.proj_embedding_hidden = None

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        # self.register_buffer('position_ids', torch.arange(max_len).expand((1, -1)))

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        """
        支持传inputs_embeds，来代替token-embedding，这个不错
        """
        if inputs_embeds is None:
            inputs_embeds = self.token_embedder_tokens(input_ids)

        # 先 project
        if self.project_embedding_first and self.proj_embedding_hidden:
            inputs_embeds = self.proj_embedding_hidden(inputs_embeds)

        bsz, length = inputs_embeds.size()[:2]
        if position_ids is None:
            position_ids = torch.arange(0, length, dtype=torch.long, device=input_ids.device).expand(bsz, length)

        position_embeddings = self.token_embedder_positions(position_ids)
        token_type_embeddings = self.token_embedder_segments(token_type_ids)
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)

        # 后 project
        if not self.project_embedding_first and self.proj_embedding_hidden:
            embeddings = self.proj_embedding_hidden(embeddings)

        return embeddings


class ALBert(nn.Module):
    """ALBert Backbone，其实是Bert的超集，比Bert多了embedding projection
    但和传统意义的albert不一样，没有实现layer共享

    name:
    fashion-albert-medium-zh:
        hdfs://haruna/home/byte_ecom_govern/user/liuyuhang/pretrain/fashion_bert_v2/config_text.yaml
        hdfs://haruna/home/byte_ecom_govern/user/liuyuhang/pretrain/fashion_bert_v2/model_state_epoch_83332.th
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.embedding = BertEmbedding(config, padding_index=2)
        self.encoder = Transformer(config)
        if self.config.with_pooler:
            self.pooler = BertPooler(config)

        # init weights
        self.apply(self.init_weights)

        if self.config.word_embedding_frozen:
            for p in self.word_embeddings.parameters():
                p.requires_grad = False

        if config.frozen_layers is not None and config.frozen_layers >= 1:
            self.frozen_parameters(config.frozen_layers)

    def init_weights(self, module):
        """Initialize the weights. # TODO: 需要吗"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, input_segment_ids, input_mask, *args, **kwargs):
        embeddings = self.embedding(
            input_ids=input_ids,
            token_type_ids=input_segment_ids,
            *args,
            **kwargs,
        )
        out = self.encoder(embeddings, input_mask)
        if isinstance(out, tuple):
            encoded_layers, attention_probs = out
        else:
            encoded_layers = out
            attention_probs = None
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output) if self.config.with_pooler else None

        return {
            "encoded_layers": encoded_layers,
            "pooled_output": pooled_output,
            "attention_probs": attention_probs,
        }
