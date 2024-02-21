# -*- coding: utf-8 -*-
import random

import numpy as np
import torch
from cruise import CruiseCLI, CruiseTrainer

# from examples.fashion_sv.data import SVDataModule
from examples.fashion_sv.dataset import SVDataModule
from examples.fashion_sv.sv_model import FashionSV

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
        FashionSV,
        trainer_class=CruiseTrainer,
        datamodule_class=SVDataModule,
        trainer_defaults={
            "summarize_model_depth": 2,
        },
    )
    cfg, trainer, model, datamodule = cli.parse_args()

    trainer.fit(model, datamodule)

# python3 examples/fashion_sv/run_train.py --config examples/fashion_sv/idx_config.yaml
