""" ALBERT + Visual """

from types import SimpleNamespace

import torch
import torch.nn as nn
import yaml

from . import albert
from .swin import SwinTransformer


class FrameALBert(nn.Module):
    """Frame + ALBert"""

    def __init__(self, config):
        """
        n层的Transformer + resnet的结构

        多帧享有一个resnet，每帧单独用resnet编码后，再过n层共享的Transformer

        如果是单独文本编码： [CLS] w1, ..., wm [SEP]
        如果是单独多帧编码： [IMG] v1, ..., vn
        如果是图文共同编码
            如果视频放前面：[IMG] v1, ..., vn [CLS] w1, ..., wm [SEP] (default)
            如果文本放前面：[CLS] w1, ..., wm [SEP] [IMG] v1, ..., vn

        文本的[CLS] 和 [SEP] 由输入来控制
        图片的[IMG] 在模型里判断，如果有多帧输入，则总是在多帧前面增加[IMG]这个token

        多帧端的embedding = [IMG]; resent embedding + position embedding。 无segment embedding

        """
        super().__init__()
        self.config = config
        self.visual_type = config.visual_type
        if self.visual_type == "SwinB224":
            self.visual = SwinTransformer(
                img_size=224,
                num_classes=512,
                embed_dim=128,
                depths=[2, 2, 18, 2],
                num_heads=[4, 8, 16, 32],
            )

        # 映射
        self.middle_size = 128  # TODO: 写死了
        self.v_projector = torch.nn.Sequential(
            torch.nn.Linear(config.visual_dim, self.middle_size),
            torch.nn.Tanh(),
            torch.nn.Linear(self.middle_size, config.embedding_size),
        )

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
        elif mode == "t":
            return self.text_only_forward(*args, **kwargs)
        elif mode == "v":
            return self.visual_only_forward(*args, **kwargs)

    def text_visual_forward(
        self,
        input_ids,
        input_segment_ids,
        input_mask,
        frames=None,
        frames_mask=None,
        visual_embeds=None,
        *args,
        **kwargs,
    ):
        """
        先两个模态一起拼接过 encoder
        如果 visual_embs 不为空，就直接用，否则会用frames 来现算
        注意：这里没有做frames为空的检查。其实不太好。
        """
        if visual_embeds is None:
            visual_embeds = self.encode_frames(frames)
        if visual_embeds.shape[-1] != self.middle_size:
            visual_embeds = self.v_projector[0](visual_embeds)
        frames_emb = self.project_frames_to_emb_size(visual_embeds)
        embeddings, m_input_mask = self.embedding(
            input_ids=input_ids,
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

    def text_only_forward(self, input_ids, input_segment_ids, input_mask, *args, **kwargs):
        """文本 only 的 forward"""
        embeddings, input_mask = self.embedding(
            input_ids=input_ids,
            token_type_ids=input_segment_ids,
            input_mask=input_mask,
            mode="t",
        )
        out = self.encoder(embeddings, input_mask)
        if isinstance(out, tuple):
            encoded_layers, attention_probs = out
        else:
            encoded_layers, attention_probs = out, None
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output) if self.config.with_pooler else None
        return {
            "encoded_layers": encoded_layers,
            "pooled_output": pooled_output,
            "embeddings": embeddings,
            "embedding_masks": input_mask,
            "attention_probs": attention_probs,
        }

    def visual_only_forward(self, frames, frames_mask, visual_embeds=None, *args, **kwargs):
        """
        frames: [bsz, frame_num, c, h, w]
        frames_mask: [bsz, frame_num]
        """
        if visual_embeds is None:
            visual_embeds = self.encode_frames(frames)
        if visual_embeds.shape[-1] != self.middle_size:
            visual_embeds = self.v_projector[0](visual_embeds)
        frames_emb = self.project_frames_to_emb_size(visual_embeds)
        embeddings, input_mask = self.embedding(visual_embeds=frames_emb, visual_mask=frames_mask, mode="v")

        out = self.encoder(embeddings, input_mask)
        if isinstance(out, tuple):
            encoded_layers, attention_probs = out
        else:
            encoded_layers, attention_probs = out, None
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output) if self.config.with_pooler else None
        return {
            "tower_output": visual_embeds,
            "frames_emb": frames_emb,
            "encoded_layers": encoded_layers,
            "pooled_output": pooled_output,
            "embeddings": embeddings,
            "embedding_masks": input_mask,
            "attention_probs": attention_probs,
        }

    def encode_frames(self, frames):
        """encode 到 128 维度"""
        N, F, C, H, W = frames.shape  # noqa: F811
        frames = torch.reshape(frames, [N * F, C, H, W])
        if self.visual_type == "RN50":
            img_feats = self.resnet(frames)["body5"]
            emb_itm = torch.mean(img_feats, dim=[-1, -2])  # 对 HW 求mean
        else:
            emb_itm = self.visual(frames)
        emb_itm = emb_itm.reshape([N, F, -1])  # [N, F, dim]
        emb = self.v_projector[0](emb_itm)
        return emb

    def project_frames_to_emb_size(self, emb):
        """
        把 128 的 frame embedding 映射到 word embedding size"""
        emb = self.v_projector[1](emb)
        emb = self.v_projector[2](emb)
        return emb


class VEmbedding(nn.Module):
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

        self.need_visual_ln = config.need_visual_ln
        if self.need_visual_ln:
            self.visual_ln = nn.LayerNorm(dim, eps=config.layernorm_eps)

        if config.embedding_size != config.hidden_size:
            self.proj_embedding_hidden = torch.nn.Linear(config.embedding_size, config.hidden_size)
        else:
            self.proj_embedding_hidden = None

        self.share_tv_pos = config.get("share_tv_pos", False)

        self.img_embedder_tokens = torch.nn.Embedding(1, dim)
        self.v_segment_embeddings = torch.nn.Embedding(1, dim)
        self.v_token_embedder_positions = torch.nn.Embedding(config.max_frame_num, dim)
        # TODO: 是否要用上面的做初始化
        # self.v_token_embedder_positions.weight = self.token_embedder_positions.weight[:config.max_frame_num]

        self.is_visual_front = config.visual_front

    def forward(
        self,
        input_ids=None,
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
        if mode == "t":
            embeddings = self.text_forward(input_ids, token_type_ids, position_ids)
        elif mode == "v":
            embeddings, input_mask = self.visual_forward(visual_embeds, visual_mask)
        elif mode == "tv":
            # 文本
            embeddings = self.text_forward(input_ids, token_type_ids, position_ids)
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

    def text_forward(self, input_ids, token_type_ids, position_ids=None):
        inputs_embeds = self.token_embedder_tokens(input_ids)
        # position
        bsz, length = inputs_embeds.size()[:2]
        if position_ids is None:
            position_ids = torch.arange(0, length, dtype=torch.long, device=input_ids.device).expand(bsz, length)
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

        if self.share_tv_pos:
            position_embeddings = self.token_embedder_positions(position_ids)
        else:
            position_embeddings = self.v_token_embedder_positions(position_ids)  # fix

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


if __name__ == "__main__":
    from collections import OrderedDict

    from cruise.utilities.cloud_io import load

    with open("easyguard/modelzoo/models/falbert/falbert.yaml") as fp:
        config = SimpleNamespace(**yaml.load(fp, yaml.Loader))
    print(config.hidden_dropout_prob)
    model = FrameALBert(config)
    print(model)
    state_dict_ori = model.state_dict()
    backbone = load(
        "hdfs://haruna/home/byte_search_nlp_lq/multimodal/modelhub/videoclip_swin_dy_20211206/model.th",
        map_location="cpu",
    )
    state_dict_new = OrderedDict()
    print(backbone.keys())
    for key, value in backbone.items():
        if key.startswith("falbert"):
            trimmed_key = key[len("falbert.") :]
        else:
            trimmed_key = key
        if trimmed_key in state_dict_ori and state_dict_ori[trimmed_key].shape == backbone[key].shape:
            state_dict_new[trimmed_key] = value
    missing_keys, unexpected_keys = model.load_state_dict(state_dict_new, strict=False)
    print("missing_keys: ", missing_keys)
    print("unexpected_keys: ", unexpected_keys)
