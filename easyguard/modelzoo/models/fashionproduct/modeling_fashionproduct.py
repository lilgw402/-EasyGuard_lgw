import os

import torch
import torch.nn as nn
from ptx.model import Model

from ...modeling_utils import ModelBase
from .swin import SwinTransformer


class FashionProduct(ModelBase):
    def __init__(
        self,
        config_text: dict,
        config_visual: dict,
        config_fusion: dict,
        config_deberta: dict,
        # load_pretrained: str = "hdfs://haruna/home/byte_ecom_govern/user/liuyuhang/pretrain/fashion_product_v2/model_state_epoch_44444.th",
        # remote_deberta_path: str = "hdfs://haruna/home/byte_ecom_govern/user/yangzheming/asr_model/zh_deberta_base_l6_emd_20210720",
        # local_deberta_dir: str = "/opt/tiger/deberta/",
        **kwargs,
    ):
        super().__init__()
        self.config_text = config_text
        self.config_visual = config_visual
        self.config_fusion = config_fusion
        self.config_deberta = config_deberta
        self.remote_deberta_path = self.config_deberta.remote_deberta_path
        self.local_deberta_dir = self.config_deberta.local_deberta_dir

        """
        Used for loading Deberta model from ptx.
        """
        suffix = self.remote_deberta_path.strip("/").split("/")[-1]
        self.local_deberta_path = f"{self.local_deberta_dir}/{suffix}"

        self.local_rank_zero_prepare()

        """
        Text Encoder
        """
        self.deberta = Model.from_option(f"file:{self.local_deberta_path}|strict=false")

        """
        Visual Encoder
        """
        self.visual = SwinTransformer(
            img_size=self.config_visual.img_size,
            num_classes=self.config_visual.output_dim,
            embed_dim=self.config_visual.embed_dim,
            depths=self.config_visual.depths,
            num_heads=self.config_visual.num_heads,
        )
        self.visual_feat = nn.Linear(self.config_visual.last_hidden_dim, self.config_visual.output_dim)
        self.visual_pos = nn.Embedding(256, self.config_visual.output_dim)
        """
        Fusion
        """
        self.visual_fuse = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.config_visual.output_dim,
                nhead=8,
                batch_first=True,
                activation="gelu",
            ),
            num_layers=1,
        )
        self.visual_pooler = nn.Sequential(
            nn.Linear(self.config_visual.output_dim, self.config_visual.output_dim),
            nn.GELU(),
        )

        """
        Calculate CLIP loss
        """
        self.t_projector = nn.Linear(self.config_text.hidden_size, self.config_fusion.hidden_size)
        self.v_projector = nn.Linear(self.config_visual.output_dim, self.config_fusion.hidden_size)

        """
        Fuse multiple text/vision fields features
        """
        self.t_projector_fuse = nn.Linear(self.config_text.hidden_size, self.config_fusion.hidden_size)
        self.v_projector_fuse = nn.Linear(self.config_visual.output_dim, self.config_fusion.hidden_size)
        self.fuse_segment_embedding = nn.Embedding(64, self.config_fusion.hidden_size)
        self.ln_text = nn.LayerNorm(self.config_fusion.hidden_size)
        self.ln_visual = nn.LayerNorm(self.config_fusion.hidden_size)
        self.ln_cls = nn.LayerNorm(self.config_fusion.hidden_size)
        self.fuse_cls = nn.Parameter(
            data=torch.randn((1, self.config_fusion.hidden_size)),
            requires_grad=True,
        )

        self.fuse_dropout = nn.Dropout(0.2)
        self.fuse = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.config_fusion.hidden_size,
                nhead=8,
                batch_first=True,
                activation="gelu",
            ),
            num_layers=self.config_fusion.num_layers,
        )
        self.fuse_pooler = nn.Linear(self.config_fusion.hidden_size, self.config_fusion.hidden_size)
        self.fuse_max_pooler = nn.AdaptiveMaxPool1d(output_size=1)

        """
        Some fixed parameters, easy for tracing.
        """
        self.PAD = 2
        self.MASK = 1
        self.SEP = 3

        self.text_segment_ids = nn.Parameter(
            data=torch.zeros((512,), dtype=torch.int64), requires_grad=False
        )  # [512, ]
        self.text_position_ids = nn.Parameter(
            data=torch.arange(0, 512, dtype=torch.int64), requires_grad=False
        )  # [512, ]self.config_fusion.hidden_size
        self.image_position_ids = nn.Parameter(
            data=torch.arange(0, 512, dtype=torch.int64), requires_grad=False
        )  # [512, ]
        # self.initialize_weights()

    def local_rank_zero_prepare(self) -> None:
        # download the tokenizer once per node
        if not os.path.exists(self.local_deberta_path):
            os.makedirs(self.local_deberta_dir, exist_ok=True)
            os.system(f"hdfs dfs -copyToLocal {self.remote_deberta_path} {self.local_deberta_path}")

    def encode_text(self, input_ids: torch.Tensor):
        batch_size, num_text, text_length = input_ids.shape
        input_ids = input_ids.view((batch_size * num_text, -1))  # [B * num_text, max_len]
        input_segment_ids = (
            self.text_segment_ids[:text_length].unsqueeze(0).expand((batch_size * num_text, -1))
        )  # [B, max_len]
        input_masks = (input_ids != self.PAD).long()  # [B, max_len]
        input_position_ids = (
            self.text_position_ids[:text_length].unsqueeze(0).expand((batch_size * num_text, -1))
        )  # [B, max_len]

        t_out = self.deberta(
            input_ids=input_ids,
            token_type_ids=input_segment_ids,
            attention_mask=input_masks,
            output_pooled=True,
        )
        t_emb, t_rep = t_out["sequence_output"], t_out["pooled_output"]

        t_emb = t_emb.view((batch_size, num_text, text_length, -1))  # [B, num_text, max_len, 768]
        t_rep = t_rep.view((batch_size, num_text, -1))  # [B, num_text, 768]

        return t_emb, t_rep

    def encode_image(self, image: torch.Tensor):
        batch_size, num_image, _, __, ___ = image.shape
        image = image.flatten(0, 1)  # [B * num_image, 3, 224, 224]

        img_out = self.visual(image, return_dict=True)
        v_emb, v_rep = (
            self.visual_feat(img_out["feature_map"]),
            img_out["pooled_out"],
        )
        v_cat = torch.cat([v_rep.unsqueeze(1), v_emb], dim=1)

        batch_size_all, image_length, _ = v_cat.shape
        position_ids = self.image_position_ids[:image_length].unsqueeze(0)
        v_cat = v_cat + self.visual_pos(position_ids)

        v_cat = self.visual_fuse(v_cat).view((batch_size, num_image, image_length, -1))
        v_rep = self.visual_pooler(v_cat[:, :, 0])

        return v_cat, v_rep

    def forward_fuse(
        self,
        main_images,
        desc_images,
        sku_images,
        main_ocrs,
        desc_ocrs,
        sku_ocrs,
        product_name,
        other_text,
        **kwargs,
    ):
        """
        :param main_images: [B, num_main_images, 3, 244, 244]
        :param desc_images: [B, num_desc_images, 3, 244, 244]
        :param sku_images: [B, num_sku_images, 3, 244, 244]
        :param main_ocrs: [B, num_main_ocrs, max_main_len]
        :param desc_ocrs: [B, num_desc_ocrs, max_desc_len]
        :param sku_ocrs: [B, num_sku_ocrs, max_sku_len]
        :param product_name: [B, max_product_len]
        :param other_text: [B, max_other_len]
        :param fuse_mask: [B, num_segments], 1 indicates valid segment, 0 indicates mask
        """

        """
        1. Encode inputs.
        """
        _, main_text_rep = self.encode_text(main_ocrs)
        _, desc_text_rep = self.encode_text(desc_ocrs)
        _, sku_text_rep = self.encode_text(sku_ocrs)
        _, product_text_rep = self.encode_text(product_name.unsqueeze(1))
        _, other_text_rep = self.encode_text(other_text.unsqueeze(1))

        _, main_image_rep = self.encode_image(main_images)
        _, desc_image_rep = self.encode_image(desc_images)
        _, sku_image_rep = self.encode_image(sku_images)

        """
        2. Modality-level clip loss
        """
        batch_size = main_text_rep.shape[0]
        fuse_cls = self.fuse_cls.unsqueeze(0).expand((batch_size, -1, -1))
        fuse_inputs_image = torch.cat(
            [
                self.ln_cls(fuse_cls + self.fuse_segment_embedding.weight[0]),
                self.ln_visual(self.v_projector_fuse(main_image_rep) + self.fuse_segment_embedding.weight[6]),
                self.ln_visual(self.v_projector_fuse(desc_image_rep) + self.fuse_segment_embedding.weight[7]),
                self.ln_visual(self.v_projector_fuse(sku_image_rep) + self.fuse_segment_embedding.weight[8]),
            ],
            dim=1,
        )
        fuse_inputs_text = torch.cat(
            [
                self.ln_cls(fuse_cls + self.fuse_segment_embedding.weight[0]),
                self.ln_text(self.t_projector_fuse(main_text_rep) + self.fuse_segment_embedding.weight[1]),
                self.ln_text(self.t_projector_fuse(desc_text_rep) + self.fuse_segment_embedding.weight[2]),
                self.ln_text(self.t_projector_fuse(sku_text_rep) + self.fuse_segment_embedding.weight[3]),
                self.ln_text(self.t_projector_fuse(product_text_rep) + self.fuse_segment_embedding.weight[4]),
                self.ln_text(self.t_projector_fuse(other_text_rep) + self.fuse_segment_embedding.weight[5]),
            ],
            dim=1,
        )
        fuse_image = self.fuse(fuse_inputs_image)[:, 0]  # [B, d_h]
        fuse_text = self.fuse(fuse_inputs_text)[:, 0]  # [B, d_h]

        """
        3. Category-level loss.
        """
        fuse_inputs = torch.cat([fuse_inputs_text, fuse_inputs_image[:, 1:]], dim=1)  # [B, 39, 512]
        fuse_inputs = self.fuse_dropout(fuse_inputs)
        fuse_emb = self.fuse(fuse_inputs)  # [B, 1 + 38, d_f]

        fuse_cls = self.fuse_pooler(fuse_emb[:, 0])  # [B, d_f]
        fuse_mp = self.fuse_pooler(self.fuse_max_pooler(fuse_emb.transpose(1, 2)).squeeze())  # [B, d_f]

        res = {
            "fuse_image": fuse_image,
            "fuse_text": fuse_text,
            "fuse_emb": fuse_emb,
            "fuse_mp": fuse_mp,
            "fuse_cls": fuse_cls,
        }

        return res
