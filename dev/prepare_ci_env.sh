#!/usr/bin/env bash

set -e

echo "====================begin to prepare ci env===================="
pip3 install -r requirements.txt
pip3 list
echo "====================finish to prepare ci env===================="