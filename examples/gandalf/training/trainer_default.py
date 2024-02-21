import os


def config_trainer(config):
    train_kwargs = config["trainer"]
    auto_resume = train_kwargs.auto_resume
    hdfs_ckpt_path = os.path.join(hdfs_output_dir, "checkpoints")
    max_steps = train_kwargs.get("max_total_iter", -1)
    if max_steps == -1:
        # default max_total_iter equals to epochs * train_max_iteration if not set
        if train_kwargs.get("epochs", 1) > 0 and train_kwargs.get("train_max_iteration", -1) > 0:
            max_steps = train_kwargs.get("epochs", 1) * train_kwargs.get("train_max_iteration", -1)
        else:
            # unable to set the max_total_iter, set to unlimited
            max_steps = -1
    # cruise_loggers = init_loggers(train_kwargs)
    cruise_loggers, callback_list = None, None
    # callback_list = self.init_callbacks(train_kwargs)
    trainer_defaults = {
        "logger": None,
        "log_every_n_steps": 50,
        "enable_versions": True,
        "precision": 16 if train_kwargs.enable_amp else 32,
        "max_epochs": train_kwargs.get("max_epochs", 1),
        "max_steps": max_steps,
        "limit_train_batches": train_kwargs.get("train_max_iteration", -1),
        "limit_val_batches": train_kwargs.get("test_max_iteration", -1),
        "val_check_interval": int(train_kwargs.get("output_iteration", 50)),
        "gradient_clip_val": train_kwargs.clip_grad_norm,
        "summarize_model_depth": 3,
        "resume_ckpt_path": hdfs_ckpt_path if auto_resume else None,
        "resume_loader_state": train_kwargs.get("resume_dataloader", False) or auto_resume,
        "project_name": "augustus",
        "experiment_name": "None",
    }
    return trainer_defaults
