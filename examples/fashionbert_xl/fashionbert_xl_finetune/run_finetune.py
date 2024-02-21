import random

import numpy as np
import torch
from cruise import CruiseCLI, CruiseTrainer

from examples.fashionbert_xl.fashionbert_xl_finetune.data import FacDataModule
from examples.fashionbert_xl.fashionbert_xl_finetune.model import FashionBertXL

rand_seed = 42


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    try:
        torch.backends.cudnn.deterministic = True
    except Exception as e:
        print(f"Exception occurred: {e}")


if __name__ == "__main__":
    # random seed
    setup_seed(rand_seed)
    cli = CruiseCLI(
        FashionBertXL,
        trainer_class=CruiseTrainer,
        datamodule_class=FacDataModule,
        trainer_defaults={
            "summarize_model_depth": 2,
        },
    )
    cfg, trainer, model, datamodule = cli.parse_args()

    trainer.fit(model, datamodule)

# python3 examples/fashionbert_xl/fashionbert_xl_finetune/run_finetune.py
# --config examples/fashionbert_xl/fashionbert_xl_finetune/default_config.yaml
