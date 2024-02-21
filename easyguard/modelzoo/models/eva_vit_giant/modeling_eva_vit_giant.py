from ...modeling_utils import ModelBase
from .eva_vit import VisionTransformer


class EvaVitGiant(ModelBase):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.vit_giant = VisionTransformer(
            img_size=config.img_size,
            patch_size=config.patch_size,
            num_classes=config.num_classes,
            use_mean_pooling=config.use_mean_pooling,
            init_values=config.init_values,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias,
            drop_path_rate=config.drop_path_rate,
            adapt=config.adapt,
            batch_size=config.batch_size,
            start_adapt_layer=config.start_adapt_layer,
            adapt_order=config.adapt_order,
            adapt_cls_dim=config.adapt_cls_dim,
            adapt_patch_dim=config.adapt_patch_dim,
            grad_checkpointing=config.grad_checkpointing,
        )

    def forward(self, image):
        rep = self.vit_giant(image)
        return rep
