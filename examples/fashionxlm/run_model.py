# -*- coding: utf-8 -*-

from cruise import CruiseCLI, CruiseTrainer

from easyguard.appzoo.sequence_classification.data import SequenceClassificationData
from easyguard.appzoo.sequence_classification.model import SequenceClassificationModel

# from easyguard.utils.arguments import print_cfg


cli = CruiseCLI(
    SequenceClassificationModel,
    trainer_class=CruiseTrainer,
    datamodule_class=SequenceClassificationData,
    trainer_defaults={
        "log_every_n_steps": 50,
        "precision": "fp16",
        "max_epochs": 1,
        "enable_versions": True,
        "val_check_interval": 1.0,  # val after 1 epoch
        "limit_val_batches": 100,
        "gradient_clip_val": 2.0,
        "sync_batchnorm": True,
        "find_unused_parameters": True,
        "summarize_model_depth": 2,
        "checkpoint_monitor": "loss",
        "checkpoint_mode": "min",
        "default_hdfs_dir": "hdfs://harunasg/home/byte_magellan_govern/users/xiaochen.qiu/roberta/",  # use your own path to save model       # noqa: E501
    },
)
cfg, trainer, model, datamodule = cli.parse_args()
# print_cfg(cfg)
trainer.fit(model, datamodule)
