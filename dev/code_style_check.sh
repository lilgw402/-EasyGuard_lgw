#!/usr/bin/env bash

set -e

echo "====================begin to check code style===================="
pip3 install -r requirements/develop.txt
pip3 list
pre-commit install
pre-commit run --all-files
echo "====================code style check passed======================"