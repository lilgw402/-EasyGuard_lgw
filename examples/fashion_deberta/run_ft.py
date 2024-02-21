"""An customizable fashion_deberta finetune example"""
from cruise import CruiseCLI, CruiseTrainer

from easyguard.appzoo.fashion_deberta.data_finetune import FashionDataFtModule
from easyguard.appzoo.fashion_deberta.model_finetune import FashionDebertaFtModel
from easyguard.utils.arguments import print_cfg

if __name__ == "__main__":
    cli = CruiseCLI(
        FashionDebertaFtModel,
        trainer_class=CruiseTrainer,
        datamodule_class=FashionDataFtModule,
        trainer_defaults={
            "max_epochs": 2,
            "val_check_interval": [3000, 1.0],
            "summarize_model_depth": 2,
            "checkpoint_monitor": "total_auc_score",
            "checkpoint_mode": "max",
            "precision": "fp16",
            "enable_versions": True,
            "default_root_dir": "/mnt/bd/yangzheming/cruise/cruise_logs",
            "default_hdfs_dir": "hdfs://haruna/home/byte_ecom_govern/user/yangzheming/chinese/common_model/trails/common_ft/asr_example/model_outputs",  # noqa: E501
        },
    )
    cfg, trainer, model, datamodule = cli.parse_args()
    print_cfg(cfg)
    trainer.fit(model, datamodule)
