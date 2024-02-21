# -*- coding: utf-8 -*-
"""
Fashionproduct_XL, feat extractor
"""
import torch

from ...modeling_utils import ModelBase
from .falbert import FrameALBert


class FashionProductXL(ModelBase):
    def __init__(
        self,
        config_backbone,
        **kwargs,
    ):
        super().__init__()
        self.config_backbone = config_backbone
        self.falbert = FrameALBert(self.config_backbone)

    def maxpooling_with_mask(self, hidden_state, attention_mask):
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).half()
        mask_expanded = 1e4 * (mask_expanded - 1)
        hidden_masked = hidden_state + mask_expanded  # sum instead of multiple
        max_pooling = torch.max(hidden_masked, dim=1)[0]

        return max_pooling

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
