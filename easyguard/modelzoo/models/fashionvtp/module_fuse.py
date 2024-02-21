""" Fusion model """
import torch
import torch.nn as nn

from easyguard.modelzoo.models.falbert import albert


class ALBertFusion(nn.Module):
    """Frame + ALBert"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        # 映射
        self.middle_size = 128  # TODO: 写死了
        self.tv_projector = torch.nn.Linear(config.hidden_size, config.embedding_size)

        # embedding
        self.embedding = VEmbedding(config, padding_index=2)
        # encoder
        self.encoder = albert.Transformer(config)
        # pooler
        if self.config.with_pooler:
            self.pooler = albert.BertPooler(config)

        self.is_visual_front = config.visual_front

        # init weights
        self.apply(self.init_weights)

    def init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, mode="tv", *args, **kwargs):
        """mode 来判断需要什么foward
        mode = tv: 视觉文本一同编码
        mode = v: 只编码视觉
        mode = t: 只编码文本
        """
        if mode == "tv":
            return self.text_visual_forward(*args, **kwargs)

    def text_visual_forward(
        self,
        input_embs,
        input_segment_ids,
        input_mask,
        frames_mask=None,
        visual_embeds=None,
        *args,
        **kwargs,
    ):
        frames_emb = self.tv_projector(visual_embeds)
        text_emb = self.tv_projector(input_embs)

        embeddings, m_input_mask = self.embedding(
            input_embs=text_emb,
            token_type_ids=input_segment_ids,
            input_mask=input_mask,
            visual_embeds=frames_emb,
            visual_mask=frames_mask,
            mode="tv",
        )

        out = self.encoder(embeddings, m_input_mask)
        if isinstance(out, tuple):
            encoded_layers, attention_probs = out
        else:
            encoded_layers = out
            attention_probs = None
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output) if self.config.with_pooler else None

        if self.is_visual_front:
            visual_length = frames_mask.shape[1]  # frame_num+1
            visual_final_out = sequence_output[:, :visual_length]
            text_final_out = sequence_output[:, visual_length:]
        else:
            text_length = input_mask.shape[1]
            text_final_out = sequence_output[:, :text_length]
            visual_final_out = sequence_output[:, text_length:]

        return {
            "encoded_layers": encoded_layers,
            "pooled_output": pooled_output,
            "text_final_output": text_final_out,  # [CLS], t1, ..., tm, [SEP]
            "visual_final_output": visual_final_out,  # [IMG], f1, ... fn
            "visual_tower_output": visual_embeds,
            "embedding_masks": m_input_mask,
            "embeddings": embeddings,
            "attention_probs": attention_probs,
        }


class VEmbedding(nn.Module):
    """
    https://code.byted.org/nlp/ptx/blob/master/ptx/core/bert.py#L340
    Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config, padding_index=2):
        super().__init__()

        self.project_embedding_first = config.project_embedding_first
        dim = config.hidden_size if self.project_embedding_first else config.embedding_size

        self.token_embedder_positions = torch.nn.Embedding(config.max_position_embeddings, dim)
        self.token_embedder_segments = torch.nn.Embedding(config.type_vocab_size, dim)

        self.norm = nn.LayerNorm(dim, eps=config.layernorm_eps)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        self.need_visual_ln = config.need_visual_ln
        if self.need_visual_ln:
            self.visual_ln = nn.LayerNorm(dim, eps=config.layernorm_eps)

        if config.embedding_size != config.hidden_size:
            self.proj_embedding_hidden = torch.nn.Linear(config.embedding_size, config.hidden_size)
        else:
            self.proj_embedding_hidden = None

        self.img_embedder_tokens = torch.nn.Embedding(1, dim)
        self.v_segment_embeddings = torch.nn.Embedding(1, dim)
        self.v_token_embedder_positions = torch.nn.Embedding(config.max_frame_num, dim)
        # TODO: 是否要用上面的做初始化
        # self.v_token_embedder_positions.weight = self.token_embedder_positions.weight[:config.max_frame_num]

        self.is_visual_front = config.visual_front

    def forward(
        self,
        input_embs=None,
        token_type_ids=None,
        position_ids=None,
        input_mask=None,
        visual_embeds=None,
        visual_mask=None,
        mode="tv",
    ):
        """
        embedding 构造。
        文本端：token embedding + position embedding + segment embedding
        视觉端：[IMG]; visual embedding + position embedding + segment embedding

        几个点:
        1. 两个模态的position embedding和 segment embedding是分开的
        2. 视觉端总是会在最开始加一个 IMG 表示整体多帧的表示
        """
        if mode == "tv":
            # 文本
            embeddings = self.text_forward(input_embs, token_type_ids, position_ids)
            # 视觉
            v_embeddings, v_input_mask = self.visual_forward(visual_embeds, visual_mask)

            if self.is_visual_front:
                embeddings = torch.cat([v_embeddings, embeddings], dim=1)
                input_mask = torch.cat([v_input_mask, input_mask], dim=1)
            else:
                embeddings = torch.cat([embeddings, v_embeddings], dim=1)
                input_mask = torch.cat([input_mask, v_input_mask], dim=1)
        else:
            raise ValueError("Unknown mode [%s] in VEmbedding forward" % mode)

        # 后处理
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        if not self.project_embedding_first and self.proj_embedding_hidden:
            embeddings = self.proj_embedding_hidden(embeddings)

        return embeddings, input_mask

    def text_forward(self, input_embs, token_type_ids, position_ids=None):
        inputs_embeds = input_embs
        # position
        bsz, length = inputs_embeds.size()[:2]
        if position_ids is None:
            position_ids = torch.arange(0, length, dtype=torch.long, device=input_embs.device).expand(bsz, length)
        position_embeddings = self.token_embedder_positions(position_ids)
        # segment
        token_type_embeddings = self.token_embedder_segments(token_type_ids)
        # 先 project
        if self.project_embedding_first and self.proj_embedding_hidden:
            inputs_embeds = self.proj_embedding_hidden(inputs_embeds)
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        return embeddings

    def visual_forward(self, visual_embeds, visual_mask, position_ids=None, *args, **kwargs):
        # 1. token
        if self.need_visual_ln:
            visual_embeds = self.visual_ln(visual_embeds)
        bsz, visual_length = visual_embeds.size()[:2]
        # 2. 纯视觉因为是没有input_ids这些，需要在最开始补一个 [IMG] 的 token
        img_embeds = self.gen_img_token_emb(bsz, visual_embeds.device)
        inputs_embeds = torch.cat([img_embeds, visual_embeds], dim=1)
        length = visual_length + 1
        # 3. mask 多加一个 [IMG] 的位置
        img_token_mask = (torch.sum(visual_mask, dim=1, keepdim=True) > 0).long()
        input_mask = torch.cat(
            [
                img_token_mask,
                visual_mask,
            ],
            dim=1,
        )
        # 4. 先 project
        if self.project_embedding_first and self.proj_embedding_hidden:
            inputs_embeds = self.proj_embedding_hidden(inputs_embeds)
        # 5. position embedding
        if position_ids is None:
            position_ids = torch.arange(0, length, dtype=torch.long, device=visual_embeds.device).expand(bsz, length)
        # position_embeddings = self.v_token_embedder_positions(position_ids) # fix
        position_embeddings = self.token_embedder_positions(position_ids)
        # 6. segment embedding
        segment_embeddings = self.v_segment_embeddings(
            torch.zeros_like(input_mask, device=input_mask.device, dtype=torch.long)
        )
        # 7. 后处理
        embeddings = inputs_embeds + position_embeddings + segment_embeddings
        return embeddings, input_mask

    def gen_img_token_emb(self, bsz, device):
        img_token = torch.zeros((bsz, 1), device=device, dtype=torch.long)
        img_embeds = self.img_embedder_tokens(img_token)
        return img_embeds
