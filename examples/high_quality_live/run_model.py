# -*- coding: utf-8 -*-
from cruise import CruiseCLI, CruiseTrainer

from easyguard.appzoo.authentic_modeling.data import AuthenticDataModule
from easyguard.appzoo.authentic_modeling.model import AuthenticMM

cli = CruiseCLI(
    AuthenticMM,
    trainer_class=CruiseTrainer,
    datamodule_class=AuthenticDataModule,
    trainer_defaults={
        "summarize_model_depth": 3,
    },
)
# pdb.set_trace()
cfg, trainer, model, datamodule = cli.parse_args()

trainer.fit(model, datamodule)
