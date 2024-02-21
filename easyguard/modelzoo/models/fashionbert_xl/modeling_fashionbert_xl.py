# -*- coding: utf-8 -*-
"""
Fashionproduct_XL, feat extractor
"""
import torch

from easyguard import AutoModel

from ...modeling_utils import ModelBase
from .swin import SwinTransformer


class FashionBertXL(ModelBase):
    def __init__(
        self,
        config_text,
        config_visual,
        config_fusion,
        **kwargs,
    ):
        super().__init__()
        self.config_text = config_text
        self.config_visual = config_visual
        self.config_fusion = config_fusion
        self.text = AutoModel.from_pretrained(self.config_text.model)  # ('/opt/tiger/xlm-roberta-base-torch')
        self.visual = SwinTransformer(
            img_size=self.config_visual.img_size,  # 224,
            num_classes=self.config_visual.hidden_dim,  # 128
            embed_dim=self.config_visual.embed_dim,  # 128
            depths=self.config_visual.depths,  # [2, 2, 18, 2],
            num_heads=self.config_visual.num_heads,  # [4, 8, 16, 32],
        )
        if self.config_fusion:
            pass

    def forward_text(
        self,
        input_ids,
        input_segment_ids,
        input_mask,
    ):
        token_pos = torch.tensor()
        text_out = self.text(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=input_segment_ids,
            position_ids=token_pos,
        )

        # text feature
        t_feat = dict()
        t_feat["t_hidden"] = text_out["last_hidden_state"]  # text_len, feat_dim
        t_feat["t_rep"] = t_feat["t_hidden"][:, 0, :]

        return t_feat

    def forward_image(
        self,
        frames=None,
        frames_mask=None,
    ):
        bz, n, c, h, w = frames.shape

        # visual feature
        v_feat = dict()
        v_feat["v_hidden"] = self.visual(frames_mask.reshape(-1, c, h, w)).reshape(bz, n, -1)  # bz, n, feat_dim
        v_feat["v_maxpool"] = self.maxpooling_with_mask(v_feat["v_hidden"], frames_mask)

        return v_feat

    def forward(
        self,
        input_ids,
        input_segment_ids,
        input_mask,
        frames=None,
        frames_mask=None,
        output_hidden=False,
    ):
        if frames_mask is None:
            frames_mask = torch.ones(frames.shape[0:2], device=frames.device, dtype=torch.long)

        mmout = self.falbert.forward(
            input_ids=input_ids,
            input_segment_ids=input_segment_ids,
            input_mask=input_mask,
            frames=frames,
            frames_mask=frames_mask,
            mode="tv",
        )

        rep_dict = dict()
        # output hidden
        if output_hidden:
            rep_dict["hidden_states"] = mmout["encoded_layers"]

        # last_hidden_state
        last_hidden_state = mmout["encoded_layers"][-1]
        rep_dict["last_hidden_state"] = last_hidden_state
        # pooler_output
        cls_emb = mmout["pooled_output"]
        rep_dict["pooler"] = cls_emb
        # max_pooling
        attention_mask = torch.cat(
            [
                input_mask,
                torch.ones([frames_mask.shape[0], 1], device=frames_mask.device),
                frames_mask,
            ],
            dim=1,
        )
        max_pooling = self.maxpooling_with_mask(hidden_state=last_hidden_state, attention_mask=attention_mask)
        rep_dict["max_pooling"] = max_pooling

        return rep_dict
