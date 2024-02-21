# -*- coding: utf-8 -*-
import os
import sys

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
try:
    import easyguard
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", ".."))

import random

import numpy as np
import torch
from cruise import CruiseCLI, CruiseTrainer

from examples.framealbert.framealbert_pretrain.data import FacDataModule
from examples.framealbert.framealbert_pretrain.model import FrameAlbertTune

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


if __name__ == "__main__":
    # random seed
    setup_seed(rand_seed)
    cli = CruiseCLI(
        FrameAlbertTune,
        trainer_class=CruiseTrainer,
        datamodule_class=FacDataModule,
        trainer_defaults={
            "summarize_model_depth": 2,
        },
    )
    cfg, trainer, model, datamodule = cli.parse_args()

    trainer.fit(model, datamodule)

# python3 examples/framealbert/framealbert_pretrain/run_train.py --config examples/framealbert/framealbert_pretrain/default_config.yaml
