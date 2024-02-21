# -*- coding: utf-8 -*-
from cruise import CruiseCLI, CruiseTrainer

from examples.fashionproduct_xl.fashionproduct_xl_audit.data import FacDataModule
from examples.fashionproduct_xl.fashionproduct_xl_audit.model import FPXL

if __name__ == "__main__":
    cli = CruiseCLI(
        FPXL,
        trainer_class=CruiseTrainer,
        datamodule_class=FacDataModule,
        trainer_defaults={},
    )
    cfg, trainer, model, datamodule = cli.parse_args()
    model.setup(stage="val")  # 初始化模型

    datamodule.setup(stage="val")  # 初始化datamodule
    trace_loader = datamodule.val_dataloader()  # 取dataloader用于生成模型输入

    checkpoint_path = ""  # 若path为空，则导出当前模型的权重，否则加载path的权重文件后再导出
    export_dir = "./traced_model"  # 指定导出路径，文件夹会自动创建

    trainer.trace(
        model_deploy=model,
        trace_dataloader=trace_loader,
        mode="anyon",  # 在此使用anyon模式导出
        checkpoint_path=checkpoint_path,
        export_dir=export_dir,
    )

# python3 examples/fashionproduct_xl/fashionproduct_xl_audit/trace.py
# --config examples/fashionproduct_xl/fashionproduct_xl_audit/default_config.yaml
