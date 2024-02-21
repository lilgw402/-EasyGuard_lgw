#!/usr/bin/env bash

set -e

echo "====================begin to test===================="
export PYTHONPATH=`pwd`:${PYTHONPATH}
# pytest参数说明: https://bytedance.feishu.cn/docx/CyuqdE43toqTUux96M9cBbpJnih
pytest -s -v -x -n auto tests/unit_tests
echo "====================tests passed======================"