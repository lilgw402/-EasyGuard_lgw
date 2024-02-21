from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....utils.losses import LearnableNTXentLoss, LearnablePCLLoss, SCELoss
from ...modeling_utils import ModelBase
from .albert import ALBert
from .swin import SwinTransformer


class FashionBert(ModelBase):
    def __init__(
        self,
        config_text: dict,
        config_visual: dict,
        config_fusion: dict,
        **kwargs,
    ):
        super().__init__()
        self.config_text = config_text
        self.config_visual = config_visual
        self.config_fusion = config_fusion

        """
        Initialize modules
        """
        self.text = ALBert(self.config_text)
        self.visual = SwinTransformer(
            img_size=self.config_visual.img_size,
            num_classes=self.config_visual.output_dim,
            embed_dim=self.config_visual.embed_dim,
            depths=self.config_visual.depths,
            num_heads=self.config_visual.num_heads,
        )
        self.visual_feat = nn.Linear(self.config_visual.last_hidden_dim, self.config_visual.output_dim)
        self.visual_pos = nn.Embedding(256, self.config_visual.output_dim)
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

        self.t_projector = nn.Linear(self.config_text.hidden_size, self.config_fusion.hidden_size)
        self.v_projector = nn.Linear(self.config_visual.output_dim, self.config_fusion.hidden_size)

        self.t_projector_fuse = nn.Linear(self.config_text.hidden_size, self.config_fusion.hidden_size)
        self.v_projector_fuse = nn.Linear(self.config_visual.output_dim, self.config_fusion.hidden_size)
        self.segment_embedding = nn.Embedding(2, self.config_fusion.hidden_size)
        self.ln_text = nn.LayerNorm(self.config_fusion.hidden_size)
        self.ln_visual = nn.LayerNorm(self.config_fusion.hidden_size)
        self.ln_cls = nn.LayerNorm(self.config_fusion.hidden_size)
        self.fuse_cls = nn.Linear(self.config_fusion.hidden_size * 2, self.config_fusion.hidden_size)

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

        # """
        # Initialize output layer
        # """
        # self.fuse_category = nn.Sequential(
        #     nn.Linear(
        #         self.config_fusion.hidden_size,
        #         self.config_fusion.hidden_size * 2,
        #     ),
        #     nn.GELU(),
        #     nn.Dropout(self.config_fusion.embd_pdrop),
        #     nn.Linear(
        #         self.config_fusion.hidden_size * 2,
        #         self.config_fusion.category_logits_level2 + 1,
        #     ),
        # )
        #
        # """
        # Initialize loss
        # """
        # self.calc_nce_loss_vt = LearnableNTXentLoss(
        #     init_tau=self.config_fusion.init_tau,
        #     clamp=self.config_fusion.tau_clamp,
        # )  # 底层图-文CLIP损失
        # self.calc_pcl_loss_ff = LearnablePCLLoss(
        #     init_tau=self.config_fusion.init_tau,
        #     clamp=self.config_fusion.tau_clamp,
        #     num_labels=self.config_fusion.category_logits_level1 + 1,
        # )  # 上层融合-融合CLIP损失，使用一级标签
        # self.category_pred_loss = SCELoss(
        #     alpha=1.0,
        #     beta=0.5,
        #     num_classes=self.config_fusion.category_logits_level2 + 1,
        # )  # 融合表征的预测损失

        """
        Initialize some fixed parameters.
        """
        self.PAD = 2
        self.MASK = 1
        self.SEP = 3
        # 文本的position ids
        self.text_position_ids = nn.Parameter(
            data=torch.arange(0, 512, dtype=torch.int64), requires_grad=False
        )  # [512, ]
        self.image_position_ids = nn.Parameter(
            data=torch.arange(0, 512, dtype=torch.int64), requires_grad=False
        )  # [512, ]
        self.text_segment_ids = nn.Parameter(
            data=torch.zeros(size=(512,), dtype=torch.int64),
            requires_grad=False,
        )  # [512, ]
        self.image_segment_ids = nn.Parameter(
            data=torch.ones(size=(512,), dtype=torch.int64),
            requires_grad=False,
        )  # [512, ]
        self.image_masks = nn.Parameter(
            data=torch.ones(size=(512,), dtype=torch.int64),
            requires_grad=False,
        )  # [512, ]
        self.clsf_masks = nn.Parameter(data=torch.ones(size=(1,), dtype=torch.int64), requires_grad=False)

        # self.initialize_weights()

    # def load_pretrained_weights(self, weight_file_path, **kwargs):
    #     state_dict_ori = self.state_dict()
    #     state_dict = torch.load(weight_file_path, map_location="cpu")
    #     state_dict_new = OrderedDict()
    #     for key, value in state_dict.items():
    #         if (
    #             key in state_dict_ori
    #             and state_dict_ori[key].shape == state_dict[key].shape
    #         ):
    #             state_dict_new[key] = value
    #     self.load_state_dict(state_dict_new, strict=False)

    def forward_text(self, token_ids: torch.Tensor):
        text_masks = (token_ids != self.PAD).long()
        text_segment_ids = (token_ids == self.SEP).long()
        text_segment_ids = torch.cumsum(text_segment_ids, dim=-1) - text_segment_ids
        text_segment_ids = torch.clamp(text_segment_ids, min=0, max=1)
        batch_size, text_length = token_ids.shape
        position_ids = self.text_position_ids[:text_length].unsqueeze(0).expand((batch_size, -1))  # [B, M]

        t_out = self.text(
            input_ids=token_ids,
            input_segment_ids=text_segment_ids,
            input_mask=text_masks,
            position_ids=position_ids,
        )
        t_emb, t_rep = t_out["encoded_layers"][-1], t_out["pooled_output"]
        t_rep = self.t_projector(t_rep)

        return t_rep, t_emb  # [B, 512], [B, 256, 768]

    def forward_image(self, image: torch.Tensor):
        img_out = self.visual(image, return_dict=True)
        v_emb, v_rep = (
            self.visual_feat(img_out["feature_map"]),
            img_out["pooled_out"],
        )

        v_cat = torch.cat([v_rep.unsqueeze(1), v_emb], dim=1)  # [B, 1 + N, d_v]
        batch_size, image_length, _ = v_cat.shape
        position_ids = self.image_position_ids[:image_length].unsqueeze(0).expand((batch_size, -1))
        v_cat = v_cat + self.visual_pos(position_ids)
        v_cat = self.visual_fuse(v_cat)  # [B, 1 + N, d_v]
        v_rep = self.visual_pooler(v_cat[:, 0])  # [B, d_v]
        v_rep = self.v_projector(v_rep)

        return v_rep, v_cat  # [B, 512], [B, 1 + 49, 512]

    def forward_fuse(self, t_emb, t_rep, text_masks, v_emb, v_rep):
        batch_size, text_length, _ = t_emb.shape
        text_segment_ids = self.text_segment_ids[:text_length].unsqueeze(0).expand((batch_size, -1))
        t_emb = self.ln_text(self.t_projector_fuse(t_emb) + self.segment_embedding(text_segment_ids))  # [B, M, d_f]

        batch_size, image_length, _ = v_emb.shape
        image_segment_ids = self.image_segment_ids[:image_length].unsqueeze(0).expand((batch_size, -1))
        v_emb = self.ln_visual(
            self.v_projector_fuse(v_emb) + self.segment_embedding(image_segment_ids)
        )  # [B, 1 + N, d_f]
        cls_emb = self.ln_cls(self.fuse_cls(torch.cat([t_rep, v_rep], dim=-1))).unsqueeze(1)  # [B, 1, d_f]
        fuse_emb = self.fuse_dropout(torch.cat([cls_emb, t_emb, v_emb], dim=1))  # [B, 1 + M + 1 + N, d_f]

        image_masks = self.image_masks[:image_length].unsqueeze(0).expand((batch_size, -1))
        cls_masks = self.clsf_masks.unsqueeze(0).expand((batch_size, -1))
        fuse_mask = (
            torch.cat([cls_masks, text_masks, image_masks], dim=1) == 0
        )  # [B, 1 + M + 1 + N], True indicates mask
        fuse_emb = self.fuse(fuse_emb, src_key_padding_mask=fuse_mask)  # [B, 1 + M + 1 + N, d_f]

        fuse_rep = self.fuse_pooler(fuse_emb[:, 0])  # [B, d_f]
        # fuse_cat = self.fuse_category(fuse_emb[:, 0])  # [B, num_categories]

        return fuse_emb, fuse_rep  # fuse_cat

    def forward(
        self,
        token_ids: torch.Tensor,
        image: torch.Tensor,
        label: torch.Tensor = None,
        **kwargs,
    ):
        text_masks = (token_ids != self.PAD).long()
        t_rep, t_emb = self.forward_text(token_ids)
        v_rep, v_emb = self.forward_image(image)
        fuse_emb, fuse_rep = self.forward_fuse(t_emb, t_rep, text_masks, v_emb, v_rep)

        return {
            "t_rep": t_rep,
            "v_rep": v_rep,
            "fuse_emb": fuse_emb,
            "fuse_rep": fuse_rep,
            # "fuse_cat": fuse_cat,
        }


class AttentionPooler(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooler, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(self, x):
        alpha = F.softmax(self.mlp(x), dim=-2)
        out = (alpha * x).sum(-2)

        return out


class AttMaxPooling(nn.Module):
    """
    使用Attention Aggregation + Max Pooling来聚合多个特征，得到一个融合特征；
    使用FC层映射到新的空间，用于对比学习；
    """

    def __init__(self, input_dim, output_dim, dropout=0.1):
        super(AttMaxPooling, self).__init__()

        self.att_pool = AttentionPooler(input_dim)
        self.max_pool = nn.AdaptiveMaxPool1d(output_size=1)

        self.fc = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x):
        """
        :param x: [B, N, d_h]
        """
        x_att = self.att_pool(x)  # [B, d_h]
        x_max = self.max_pool(x.transpose(1, 2)).squeeze()  # [B, d_h]
        x_out = self.fc(torch.cat([x_att, x_max], dim=-1))

        return x_out
