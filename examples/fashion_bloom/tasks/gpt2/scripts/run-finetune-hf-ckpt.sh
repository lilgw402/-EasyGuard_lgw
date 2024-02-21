#!/bin/bash
set -x

declare -A model_dir
model_dir=(
    ['bloom_7b1']='hdfs://haruna/home/byte_ecom_govern/user/doushihan/hf_models/bloom-7b1'
    ['bloom_560m']='hdfs://haruna/home/byte_ecom_govern/user/doushihan/hf_models/bloom560m/bloomz-560m'
    ['bloom_7b1_mnt']='/mnt/bn/ecom-govern-maxiangqian/doushihan/hf_models/bloom7b1/bloom-7b1'
    ['bloom_560m_mnt']='/mnt/bn/ecom-govern-maxiangqian/doushihan/hf_models/bloom560m/bloomz-560m'
)

trainer_config=(
    ['bloom_7b1']='tasks/gpt2/zero_shot_eval/zero2-fp16.yaml'
    ['bloom_560m']='tasks/gpt2/zero_shot_eval/zero2-fp16.yaml'
    ['bloom_7b1_mnt']='tasks/gpt2/zero_shot_eval/zero2-fp16.yaml'
    ['bloom_560m_mnt']='tasks/gpt2/zero_shot_eval/zero2-fp16.yaml'
)

chkpt_path=${model_dir[$@]}
yaml_path=${trainer_config[$@]}

bash launch.sh tasks/gpt2/zero_shot_eval/model.py \
  --model.use_hf_ckpt=True \
  --data.from_hf_tokenizer=True \
  --data.tokenizer="$chkpt_path" \
  --data.hf_tokenizer_use_fast=False \
  --data.max_seq_len=1024 \
  --model=tasks/gpt2/zero_shot_eval/1b.yaml \
  --model.partial_pretrain="$chkpt_path" \
  --model.model_config="$chkpt_path" \
  --data.train_num_workers=1 \
  --data.train_batch_size=2 \
  --data.val_num_workers=1 \
  --data.val_batch_size=2 \
  --trainer.val_check_interval=0.5 \
  --data.train_path=/mnt/bn/ecom-govern-maxiangqian/doushihan/data/junjun_v1/train_0130_add_trans2_prompt.parquet \
  --data.val_path=--data.val_path=/mnt/bn/ecom-govern-maxiangqian/doushihan/data/junjun_v1/test_0130_add_trans2_prompt.parquet \
  --data.dataset_name=lambada \
  --data.template_name=please+next+word  \
  --data.finetune_type_is_qa=true \
  --data.drop_last=True \
  --model.network.use_rmpad_lmloss=false \
  --model.network.use_rmpad_lnmlp=false \
  --model.network.use_rmpad_attn=false \
  --model.network.pad_idx=3 \
  --trainer.max_epochs=3 \
  --trainer.optimizer_kwargs.optimizer.params.lr=1e-5 \
  --trainer.default_hdfs_dir=/mnt/bn/ecom-govern-maxiangqian/doushihan/finetune_models/bloom/debug \
  --trainer="$yaml_path"