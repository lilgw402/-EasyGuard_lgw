bash launch.sh tasks/gpt2/zero_shot_eval/model.py \
  --model.use_hf_ckpt=False \
  --data.from_hf_tokenizer=True \
  --data.tokenizer=/mnt/bn/ecom-ccr-dev/mlx/users/doushihan/hf_models/bloomz-7b1 \
  --data.hf_tokenizer_use_fast=False \
  --generate-steps=1024 \
  --model=tasks/gpt2/zero_shot_eval/1b.yaml \
  --model.partial_pretrain=hdfs://harunava/home/byte_magellan_va/user/doushihan/models/bloom7b1-chatcat-bsz4-ga4-wp-60k-v2-0407/checkpoints/global_step_1207/mp_rank_00_model_states.pt \
  --model.model_config=/mnt/bn/ecom-ccr-dev/mlx/users/doushihan/hf_models/bloomz-7b1 \
  --model.network.pad_idx=3 \
  --play \
  --generate-temp 1.3




bash launch.sh tasks/gpt2/zero_shot_eval/model.py \
  --model.use_hf_ckpt=False \
  --data.from_hf_tokenizer=True \
  --data.tokenizer=/mnt/bn/ecom-ccr-dev/mlx/users/doushihan/hf_models/bloomz-7b1 \
  --data.hf_tokenizer_use_fast=False \
  --data.max_seq_len=128 \
  --generate-steps=256 \
  --model=tasks/gpt2/zero_shot_eval/1b.yaml \
  --model.partial_pretrain=/mnt/bn/ecom-ccr-dev/mlx/users/doushihan/finetune_models/bloom/bloom7b1-chatcat-bsz4-warmup-1n8g/checkpoints/global_step_2295/mp_rank_00_model_states.pt \
  --model.model_config=/mnt/bn/ecom-ccr-dev/mlx/users/doushihan/hf_models/bloomz-7b1 \
  --model.network.pad_idx=3 \
  --generate-temp 0.7 \
  --trace-only