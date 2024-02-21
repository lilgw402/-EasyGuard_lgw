#!/bin/bash
ROOT_HOME=`pwd -P`
echo 'Curdir: '$ROOT_HOME
project_dir="$(cd examples/gandalf/; pwd)"
echo 'ProjectDir: '$project_dir
cd $project_dir
cur_dir=`pwd -P`
echo 'Curdir: '$cur_dir
echo 'GPU NUM: '$ARNOLD_WORKER_GPU

export PYTHONPATH=$PYTHONPATH:/opt/tiger
export PYTHONPATH=$PYTHONPATH:/opt/tiger/anyon2
export PYTHONPATH=$PYTHONPATH:/opt/tiger/titan
export PYTHONPATH=$PYTHONPATH:/opt/tiger/cruise
export PYTHONPATH=$PYTHONPATH:$ROOT_HOME


# # pretrain weights for custom tokenizer
echo 'pulling down pretrained weights...'
if [ ! -d "models/weights/simcse_bert_base" ]; then mkdir -p models/weights/simcse_bert_base; fi
if [ ! -d "models/weights/simcse_bert_base" ]; then hadoop fs -get  hdfs:///user/jiangxubin/models/pretrain/simcse_bert_base ./models/weights/simcse_bert_base; fi
if [ ! -d "models/weights/fashion_deberta_asr/" ]; then mkdir -p models/weights/fashion_deberta_asr/; fi
if [ ! -d "models/weights/fashion_deberta_asr/deberta_3l" ]; then hadoop fs -get  hdfs:///user/jiangxubin/models/pretrain/fashion_deberta_asr/deberta_3l ./models/weights/fashion_deberta_asr; fi

# # trigger train proc
export ROOT_HOME=${ROOT_HOME:=$ROOT_HOME}
export PROJECT_HOME=${PROJECT_HOME:=$project_dir}
export CRUISE_HOME=${CRUISE_HOME:=/opt/tiger/cruise}

echo 'ROOT_HOME: '$ROOT_HOME
echo 'PROJECT_HOME: '$PROJECT_HOME
echo 'CRUISE_HOME: '$CRUISE_HOME

if [ -f "$CRUISE_HOME/cruise/tools/TORCHRUN" ]; then
  $CRUISE_HOME/cruise/tools/TORCHRUN  $PROJECT_HOME/main.py $@
else
  echo 'CRUISE_HOME not exist'
fi