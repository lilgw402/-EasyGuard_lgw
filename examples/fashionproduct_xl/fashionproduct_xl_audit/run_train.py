import random

import numpy as np
import torch
from cruise import CruiseCLI, CruiseTrainer

from examples.fashionproduct_xl.fashionproduct_xl_audit.data import FacDataModule
from examples.fashionproduct_xl.fashionproduct_xl_audit.model import FPXL

rand_seed = 42


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    try:
        torch.backends.cudnn.deterministic = True
    except Exception as e:
        print(f"exception occurred: {e}")


if __name__ == "__main__":
    # random seed
    setup_seed(rand_seed)
    cli = CruiseCLI(
        FPXL,
        trainer_class=CruiseTrainer,
        datamodule_class=FacDataModule,
        trainer_defaults={
            "summarize_model_depth": 2,
        },
    )
    cfg, trainer, model, datamodule = cli.parse_args()

    trainer.fit(model, datamodule)

# python3 examples/fashionproduct_xl/fashionproduct_xl_audit/run_train.py
# --config examples/fashionproduct_xl/fashionproduct_xl_audit/default_config.yaml
