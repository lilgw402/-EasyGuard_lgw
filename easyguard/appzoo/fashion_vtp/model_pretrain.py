# -*- coding: utf-8 -*-
"""
FashionVTP Model
"""
import math
import os.path
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

try:
    import cruise
except ImportError:
    print(
        "[ERROR] cruise is not installed! Please refer this doc: https://bytedance.feishu.cn/wiki/wikcnGP7yzZAuKpPfL6jRJKl2ag"
    )

from cruise import CruiseModule
from cruise.utilities.cloud_io import load
from cruise.utilities.hdfs_io import hexists, hopen

from .falbert import FrameALBert
from .module_fuse import ALBertFusion


class FashionVTP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.falbert = FrameALBert(config)
        self.fusemodel = ALBertFusion(config)
        self.t_projector, self.v_projector = self.init_projector()
        # add momentum model
        self.falbert_m = FrameALBert(config)
        self.t_projector_m, self.v_projector_m = self.init_projector()

    """
    def forward(
        self,
        images: torch.Tensor,
        images_mask: torch.Tensor,
        input_ids: torch.Tensor,
        input_mask: torch.Tensor,
        input_segment_ids: torch.Tensor = None,
        *args,
        **kwargs):

        t_output = self.encode_text(input_ids=input_ids,
                                    input_segment_ids=input_segment_ids,
                                    input_mask=input_mask)
        t_emb = t_output['pooled_output']
        t_emb = self.t_projector(t_emb)
        t_tokens = t_output['encoded_layers'][-1]

        v_output = self.encode_image(images=images, images_mask=images_mask)
        v_emb = v_output['pooled_output']
        v_emb = self.v_projector(v_emb)
        v_tokens = v_output['encoded_layers'][-1][:, 1:, :]

        mmout =  self.fusemodel(
                        input_embs=t_tokens,
                        input_segment_ids=input_segment_ids,
                        input_mask=input_mask,
                        frames_mask=images_mask,
                        visual_embeds=v_tokens)

        mm_emb = mmout['pooled_output']
        return mm_emb, t_emb, v_emb
    """

    def encode_image(
        self,
        images: torch.Tensor,
        images_mask: torch.Tensor = None,
        visual_embeds: torch.tensor = None,
    ):
        if len(images.shape) == 4:
            images = images.unsqueeze(1)
        if images_mask is None:
            if visual_embeds is None:
                images_mask = torch.ones(images.shape[0:2], device=images.device, dtype=torch.long)
            else:
                images_mask = torch.ones(
                    visual_embeds.shape[0:2],
                    device=visual_embeds.device,
                    dtype=torch.long,
                )
        v_out = self.falbert(
            frames=images,
            frames_mask=images_mask,
            visual_embeds=visual_embeds,
            mode="v",
        )
        return v_out

    def encode_text(
        self,
        input_ids: torch.Tensor,
        input_mask: torch.Tensor,
        input_segment_ids: torch.Tensor = None,
    ):
        if input_segment_ids is None:
            input_segment_ids = torch.zeros_like(input_ids, device=input_ids.device)
        t_out = self.falbert(
            input_ids=input_ids,
            input_segment_ids=input_segment_ids,
            input_mask=input_mask,
            mode="t",
        )
        return t_out

    def encode_image_m(
        self,
        images: torch.Tensor,
        images_mask: torch.Tensor = None,
        visual_embeds: torch.tensor = None,
    ):
        if len(images.shape) == 4:
            images = images.unsqueeze(1)
        if images_mask is None:
            if visual_embeds is None:
                images_mask = torch.ones(images.shape[0:2], device=images.device, dtype=torch.long)
            else:
                images_mask = torch.ones(
                    visual_embeds.shape[0:2],
                    device=visual_embeds.device,
                    dtype=torch.long,
                )
        v_out = self.falbert_m(
            frames=images,
            frames_mask=images_mask,
            visual_embeds=visual_embeds,
            mode="v",
        )
        return v_out

    def encode_text_m(
        self,
        input_ids: torch.Tensor,
        input_mask: torch.Tensor,
        input_segment_ids: torch.Tensor = None,
    ):
        if input_segment_ids is None:
            input_segment_ids = torch.zeros_like(input_ids, device=input_ids.device)
        t_out = self.falbert_m(
            input_ids=input_ids,
            input_segment_ids=input_segment_ids,
            input_mask=input_mask,
            mode="t",
        )
        return t_out

    # def encode_multimodal(
    #     self,
    #     input_ids,
    #     input_segment_ids,
    #     input_mask,
    #     images=None,
    #     images_mask=None,
    #     visual_embeds=None,
    #     *args,
    #     **kwargs,
    # ):
    #     if images_mask is None:
    #         if visual_embeds is None:
    #             images_mask = torch.ones(
    #                 images.shape[0:2], device=images.device, dtype=torch.long
    #             )
    #         else:
    #             images_mask = torch.ones(
    #                 visual_embeds.shape[0:2],
    #                 device=visual_embeds.device,
    #                 dtype=torch.long,
    #             )
    #     mmout = self.falbert(
    #         input_ids=input_ids,
    #         input_segment_ids=input_segment_ids,
    #         input_mask=input_mask,
    #         frames=images,
    #         frames_mask=images_mask,
    #         visual_embeds=visual_embeds,
    #         mode="tv",
    #     )
    #     return mmout

    def init_projector(
        self,
        input_size=768,
        output_size=128,
    ):
        v_projector = torch.nn.Linear(input_size, output_size)
        t_projector = torch.nn.Linear(input_size, output_size)
        return t_projector, v_projector
