# -*- coding: utf-8 -*-
from cruise import CruiseCLI, CruiseTrainer

from examples.fashionproduct_xl.fashionproduct_xl_audit.data import FacDataModule
from examples.fashionproduct_xl.fashionproduct_xl_audit.model import FPXL

if __name__ == "__main__":
    cli = CruiseCLI(
        FPXL,
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

# 2023-05-21 04:12:48.407938 finished: ID_1013.test.jsonl -- 40596/40597
# 2023-05-21 04:15:02.326056 finished: VN_1013.test.jsonl -- 39824/39852
# 2023-05-21 04:20:47.956048 finished: TH_1013.test.jsonl -- 43792/43811
# 2023-05-21 04:21:57.920726 finished: MY_1013.test.jsonl -- 43926/43962
# 2023-05-21 04:30:11.792076 finished: GB_1013.test.jsonl -- 83331/83335
