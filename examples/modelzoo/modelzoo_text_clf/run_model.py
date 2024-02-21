# -*- coding: utf-8 -*-
import os
import sys

try:
    import easyguard
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cruise import CruiseCLI, CruiseTrainer
from data import SequenceClassificationData
from model import ModelZooTextClf

from easyguard.utils.arguments import print_cfg

cli = CruiseCLI(
    ModelZooTextClf,
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
        "default_hdfs_dir": "hdfs://harunasg/home/byte_magellan_govern/users/xiaochen.qiu/modelzoo/",  # use your own path to save model
    },
)
cfg, trainer, model, datamodule = cli.parse_args()
print_cfg(cfg)
trainer.fit(model, datamodule)
