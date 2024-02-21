#!/usr/bin/env bash

set -ex
pip3 uninstall numpy -y
pip3 install numpy==1.23.4
pip3 install jinja2
pip3 install https://luban-source.byted.org/repository/scm/data.aml.xperf_gpt_th20_cu117_abi0_sdist_1.0.0.83.tar.gz
# 提前把模型上传到HDFS上
MODEL_PATH=hdfs://haruna/home/byte_ecom_govern/easyguard/hukongtao/Bernard/models/chinese-alpaca-2-13b-1008
env /opt/tiger/arnold/hdfs_client/hdfs dfs -get "${MODEL_PATH}" -s -c 512 --ct 32 -t 8 .
pip3 list