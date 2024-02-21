#!/bin/bash
set -x

declare -A model_dir
model_dir=(
    ['bloom_7b1']='hdfs://haruna/home/byte_ecom_govern/user/doushihan/hf_models/bloom-7b1'
    ['bloom_560m']='hdfs://haruna/home/byte_ecom_govern/user/doushihan/hf_models/bloom560m/bloomz-560m'
    ['bloom_7b1_mnt']='/mnt/bn/ecom-govern-maxiangqian/doushihan/hf_models/bloom7b1/bloom-7b1'
    ['bloom_560m_mnt']='/mnt/bn/ecom-govern-maxiangqian/doushihan/hf_models/bloom560m/bloomz-560m'
)

chkpt_path=${model_dir[$@]}

bash launch.sh tasks/gpt2/zero_shot_eval/model.py \
  --model.use_hf_ckpt=True \
  --data.from_hf_tokenizer=True \
  --data.tokenizer="$chkpt_path" \
  --data.hf_tokenizer_use_fast=False \
  --data.max_seq_len=512 \
  --model=tasks/gpt2/zero_shot_eval/1b.yaml \
  --model.partial_pretrain="$chkpt_path" \
  --model.model_config="$chkpt_path" \
  --data.train_num_workers=1 \
  --data.train_batch_size=1 \
  --data.val_num_workers=1 \
  --data.val_batch_size=1 \
  --trainer.val_check_interval=0.5 \
  --data.train_path=hdfs://haruna/home/byte_ecom_govern/user/doushihan/data/lambada/dev_4864/dev.parquet  \
  --data.val_path=hdfs://haruna/home/byte_ecom_govern/user/doushihan/data/lambada/clean_test \
  --data.dataset_name=lambada \
  --data.template_name=please+next+word  \
  --data.drop_last=True \
  --model.network.use_rmpad_lmloss=false \
  --model.network.use_rmpad_lnmlp=false \
  --model.network.use_rmpad_attn=false \
  --model.network.pad_idx=3 \
  --trainer.max_epochs=20 \
  --trainer.optimizer_kwargs.optimizer.params.lr=1e-5 \
  --play-file-type="qa" \
  --play-file /mnt/bn/ecom-govern-maxiangqian/doushihan/data/junjun_v1/test_0130_add_trans2_prompt.parquet \
  --play-out-file hdfs://haruna/home/byte_ecom_govern/user/doushihan/jj_output/hf_bloom560m_jjtest_output.jsonl \
  --generate-temp 0.7

