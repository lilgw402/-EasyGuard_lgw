# -*- coding:utf-8 -*-
# Email:    jiangxubin@bytedance.com
# Created:  2023-03-01 12:52:19
# Modified: 2023-03-01 12:52:19
import argparse
import os
import re
import sys

from addict import Dict
from cruise import CruiseCLI, CruiseTrainer
from utils.config import config
from utils.driver import DIST_CONTEXT, init_device
from utils.file_util import check_hdfs_exist, hmkdir
from utils.registry import get_data_module, get_model_module
from utils.util import init_seeds, load_conf, load_from_yaml

from easyguard.utils.arguments import print_cfg


def prepare_folder(config):
    os.makedirs(config.trainer.default_root_dir, exist_ok=True)
    if not check_hdfs_exist(config.trainer.default_hdfs_dir):
        hmkdir(config.trainer.default_hdfs_dir)


def prepare_gandalf_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="local yaml conf")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--val", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/mnt/bn/renaisance/mlx/data/cruise_logs/gandalf/exps/version_1/checkpoints/epoch=0-step=1000-loss=0.655.ckpt",
    )
    parser.add_argument(
        "--export_dir",
        type=str,
        default="/mnt/bn/renaisance/mlx/models/serving/gandalf/cruise/base",
    )
    args, config_override = parser.parse_known_args()
    sys.argv = [
        arg for arg in sys.argv if len(re.findall("--fit|--val|--trace|--checkpoint_path|--export_dir", arg)) == 0
    ]
    if args.config:
        config = Dict(load_from_yaml(args.config))
    else:
        raise FileNotFoundError("config file option must be specified")
    config.update({"fit": args.fit, "val": args.val, "trace": args.trace})
    config.update(
        {
            "checkpoint_path": args.checkpoint_path,
            "export_dir": args.export_dir,
        }
    )
    return Dict(config)


def prepare_common_trainer_defaults(config):
    trainer_defaults = {
        "seed": 42,
        "precision": 16,
        "enable_checkpoint": (30, 30),
        "resume_ckpt_path": "auto",
        "summarize_model_depth": 5,
        "checkpoint_monitor": "loss",
        "checkpoint_mode": "min",
    }
    return trainer_defaults


def prepare_trainer_components(config):
    model_module = get_model_module(config["model"]["type"])
    data_module = get_data_module(config["data"]["type"])
    return model_module, data_module


def main():
    config = prepare_gandalf_args()
    init_seeds(42 + DIST_CONTEXT.global_rank, cuda_deterministic=True)
    model_module, data_module = prepare_trainer_components(config)
    trainer_defaults = prepare_common_trainer_defaults(config)
    cli = CruiseCLI(
        model_module,
        data_module,
        trainer_class=CruiseTrainer,
        trainer_defaults=trainer_defaults,
    )
    cfg, trainer, model, datamodule = cli.parse_args()
    print_cfg(cfg)
    if config["fit"]:
        trainer.fit(model, datamodule=datamodule)
    if config["val"]:
        trainer.validate(model, datamodule=datamodule)
    if config["trace"]:
        model.setup("val")
        datamodule.setup("val")
        print("===Trace Begin===")
        # checkpoint_path = "/mnt/bn/renaisance/mlx/data/cruise_logs/gandalf/exps/version_1/checkpoints/epoch=0-step=1000-loss=0.655.ckpt"
        # export_dir = "/mnt/bn/renaisance/mlx/models/serving/gandalf/cruise/base"
        trainer.trace(
            model_deploy=model,
            trace_dataloader=datamodule.val_dataloader(),
            mode="anyon",
            checkpoint_path=config.checkpoint_path,
            export_dir=config.export_dir,
        )


if __name__ == "__main__":
    main()
