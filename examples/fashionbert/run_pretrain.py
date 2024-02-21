# -*- coding: utf-8 -*-
from cruise import CruiseCLI, CruiseTrainer

from easyguard.appzoo.multimodal_modeling.data import MMDataModule
from easyguard.appzoo.multimodal_modeling.model import FashionBertv2
from easyguard.utils.arguments import print_cfg

if __name__ == "__main__":
    cli = CruiseCLI(
        FashionBertv2,
        trainer_class=CruiseTrainer,
        datamodule_class=MMDataModule,
        trainer_defaults={
            "log_every_n_steps": 100,
            "precision": "fp16",
            "max_epochs": 2,
            "enable_versions": True,
            "val_check_interval": 500,  # val after 1 epoch
            "limit_val_batches": 100,
            "gradient_clip_val": 2.0,
            "sync_batchnorm": True,
            "find_unused_parameters": True,
            "summarize_model_depth": 2,
            "checkpoint_monitor": "loss",
            "checkpoint_mode": "min",
            "default_hdfs_dir": "hdfs://haruna/home/byte_ecom_govern/user/liuyuhang/pretrain/checkpoint_pt",
        },
    )
    cfg, trainer, model, datamodule = cli.parse_args()
    print_cfg(cfg)
    trainer.fit(model, datamodule)
