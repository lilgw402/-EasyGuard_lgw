# -*- coding: utf-8 -*-

import random

import numpy as np
import torch
from cruise import CruiseCLI, CruiseTrainer

from easyguard.appzoo.high_quality_live.data import HighQualityLiveDataModule
from easyguard.appzoo.high_quality_live.model_videoclip import HighQualityLiveVideoCLIP

rand_seed = 42


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    try:
        torch.backends.cudnn.deterministic = True
    except:  # noqa: E722
        ...


# random seed
setup_seed(rand_seed)

cli = CruiseCLI(
    HighQualityLiveVideoCLIP,
    trainer_class=CruiseTrainer,
    datamodule_class=HighQualityLiveDataModule,
    trainer_defaults={
        "summarize_model_depth": 3,
    },
)

cfg, trainer, model, datamodule = cli.parse_args()

trainer.fit(model, datamodule)
