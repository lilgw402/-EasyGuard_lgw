#!/usr/bin/env bash
# python3 /mnt/bn/ecom-govern-maxiangqian/qingxuan/EasyGuard/examples/high_quality_live/run_model.py --config /mnt/bn/ecom-govern-maxiangqian/qingxuan/EasyGuard/examples/high_quality_live/config/hq_config_baseline.yaml
# cd /mnt/bn/ecom-govern-maxiangqian/dongjunwei/EasyGuard
# tools/TORCHRUN examples/high_quality_live/run_model_videoclip.py  \
#     --config examples/high_quality_live/config/videoclip/train_videoclip.yaml 

python3 -m debugpy --connect $(hostname -i | awk '{print $2}'):6032 examples/high_quality_live/run_model_videoclip.py  \
    --config examples/high_quality_live/config/videoclip/train_videoclip.yaml 