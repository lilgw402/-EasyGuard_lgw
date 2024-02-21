#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# add lucifer to PYTHONPATH, also add cruise to PYTHONPATH if using scm
export PYTHONPATH=$SCRIPT_DIR:/opt/tiger/cruise:$PYTHONPATH

if [[ -z "$ARNOLD_DEBUG" ]]; then
  bash setup_cruise.sh
fi

# show all configs with defaults and user override first
python3 $@ --print_config

# run it
echo "xxxxx about to launch"
date
if ! command -v TORCHRUN &> /dev/null
then
    echo "==<TORCHRUN could not be found, use cruise included script>=="
    /opt/tiger/cruise/cruise/tools/TORCHRUN $@
    exit
else
    TORCHRUN $@
fi
