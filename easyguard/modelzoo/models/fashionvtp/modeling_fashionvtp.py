import torch

from ...modeling_utils import ModelBase
from .falbert import FrameALBert
from .module_fuse import ALBertFusion


class FashionVTP(ModelBase):
    def __init__(
        self,
        config_model: dict,
        **kwargs,
    ):
        super().__init__()

        self.config_model = config_model
        self.falbert = FrameALBert(self.config_model)
        self.fusemodel = ALBertFusion(self.config_model)
        self.t_projector, self.v_projector = self.init_projector()

    def forward(
        self,
        images: torch.Tensor,
        images_mask: torch.Tensor,
        input_ids: torch.Tensor,
        input_mask: torch.Tensor,
        input_segment_ids: torch.Tensor = None,
        *args,
        **kwargs,
    ):
        t_output = self.encode_text(
            input_ids=input_ids,
            input_segment_ids=input_segment_ids,
            input_mask=input_mask,
        )
        t_emb = t_output["pooled_output"]
        t_emb = self.t_projector(t_emb)
        t_tokens = t_output["encoded_layers"][-1]

        v_output = self.encode_image(images=images, images_mask=images_mask)
        v_emb = v_output["pooled_output"]
        v_emb = self.v_projector(v_emb)
        v_tokens = v_output["encoded_layers"][-1][:, 1:, :]

        mmout = self.fusemodel(
            input_embs=t_tokens,
            input_segment_ids=input_segment_ids,
            input_mask=input_mask,
            frames_mask=images_mask,
            visual_embeds=v_tokens,
        )

        mm_emb = mmout["pooled_output"]
        return mm_emb, t_emb, v_emb

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

    def init_projector(
        self,
        input_size=768,
        output_size=128,
    ):
        v_projector = torch.nn.Linear(input_size, output_size)
        t_projector = torch.nn.Linear(input_size, output_size)
        return t_projector, v_projector

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
