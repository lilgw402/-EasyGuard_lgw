# -*- coding: utf-8 -*-
from cruise import CruiseCLI, CruiseTrainer

from examples.fashionbert_xl.fashionbert_xl_finetune.data import FacDataModule
from examples.fashionbert_xl.fashionbert_xl_finetune.model import FashionBertXL

if __name__ == "__main__":
    cli = CruiseCLI(
        FashionBertXL,
        trainer_class=CruiseTrainer,
        datamodule_class=FacDataModule,
        trainer_defaults={
            "summarize_model_depth": 2,
        },
    )
    cfg, trainer, model, datamodule = cli.parse_args()

    # model.setup(stage="val")        # 初始化模型

    datamodule.setup(stage="val")  # 初始化datamodule
    eval_loader = datamodule.val_dataloader()  # 取dataloader用于生成模型输入

    trainer.validate(model=model, val_dataloader=eval_loader)

# python3 examples/fashionbert_xl/fashionbert_xl_finetune/eval.py
# --config examples/fashionbert_xl/fashionbert_xl_finetune/eval.yaml
