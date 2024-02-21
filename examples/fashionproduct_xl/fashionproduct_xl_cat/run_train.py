# -*- coding: utf-8 -*-

import random

import numpy as np
import torch
from cruise import CruiseCLI, CruiseTrainer

from examples.fashionproduct_xl.fashionproduct_xl_cat.data import FacDataModule
from examples.fashionproduct_xl.fashionproduct_xl_cat.model import FrameAlbertClassify

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


if __name__ == "__main__":
    # random seed
    setup_seed(rand_seed)
    cli = CruiseCLI(
        FrameAlbertClassify,
        trainer_class=CruiseTrainer,
        datamodule_class=FacDataModule,
        trainer_defaults={
            "summarize_model_depth": 2,
        },
    )
    cfg, trainer, model, datamodule = cli.parse_args()

    trainer.fit(model, datamodule)

# python3 examples/fashionproduct_xl/fashionproduct_xl_cat/run_train.py
#     --config examples/fashionproduct_xl/fashionproduct_xl_cat/default_config.yaml
