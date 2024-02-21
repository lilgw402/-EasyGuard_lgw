"""Convenient optimizer settings used in mariana"""
from cruise import CruiseConfig

mariana_optimizer_kwargs_defaults = CruiseConfig(
    {
        "optimizer": {
            "type": "deepspeed.ops.adam.FusedAdam",
            "params": {
                "lr": 5e-5,
                "betas": [0.9, 0.999],
                "eps": 1.0e-08,
                "weight_decay": 0.01,
                "bias_correction": True,
                "adam_w_mode": True,
            },
        },
        "scheduler": {
            "type": "mariana.optim.lr_scheduler.get_linear_schedule_with_warmup",
            "total_steps_param_name": "num_training_steps",
            "warmup_steps_param_name": "num_warmup_steps",
            "interval": "step",
            "params": {
                "warmup_step_rate": 0.005,
                "lr_end": 1e-7,
            },
        },
    }
)
