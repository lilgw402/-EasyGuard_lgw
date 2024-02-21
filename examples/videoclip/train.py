# -*- coding: utf-8 -*-
import os
import sys

try:
    import easyguard
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

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
    except:
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
