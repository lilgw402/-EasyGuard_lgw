bash launch.sh tasks/gpt2/model.py \
  --model.use_hf_ckpt=False \
  --data.from_hf_tokenizer=True \
  --data.tokenizer=/mnt/bn/ecom-ccr-dev-878d5f0f/mlx/users/doushihan/hf_models/bloomz-7b1 \
  --data.hf_tokenizer_use_fast=False \
  --data.max_seq_len=300 \
  --model=tasks/gpt2/1b.yaml \
  --model.partial_pretrain=/mnt/bn/ecom-ccr-dev-878d5f0f/mlx/users/doushihan/finetune_models/bloom/bloom7b1-instruct-tuning-ccr71-0329/checkpoints/global_step_9795/mp_rank_00_model_states.pt \
  --model.model_config=/mnt/bn/ecom-ccr-dev-878d5f0f/mlx/users/doushihan/hf_models/bloomz-7b1 \
  --model.network.pad_idx=3 \
  --play-file-type="qa_batch" \
  --play-file-bsz 40 \
  --generate-steps 30 \
  --play-file /mnt/bn/ecom-ccr-dev-878d5f0f/mlx/users/doushihan/data/ccr_jj_cls71/test_0130_add_trans2_prompt2.parquet \
  --play-out-file /mnt/bn/ecom-ccr-dev-878d5f0f/mlx/users/doushihan/playfile_output/bloom7b1_instruct_tuning_test_0130_add_trans2_prompt2_output.jsonl \
  --generate-temp 0.7


/mnt/bn/ecom-ccr-dev/mlx/users/doushihan/finetune_models/bloom/bloo1b1_0316_100B_round3_nowp_with_ga_round2/checkpoints/global_step_5000/mp_rank_00_model_states.pt

bash launch.sh tasks/gpt2/model.py \
  --model.use_hf_ckpt=False \
  --data.from_hf_tokenizer=True \
  --data.tokenizer=/mnt/bn/ecom-ccr-dev/mlx/users/doushihan/hf_models/bloomz-1b1 \
  --data.hf_tokenizer_use_fast=False \
  --data.max_seq_len=300 \
  --model=tasks/gpt2/1b.yaml \
  --model.partial_pretrain=/mnt/bn/ecom-ccr-dev/mlx/users/doushihan/finetune_models/bloom/bloo1b1_0316_100B_round3_nowp_with_ga_round2/checkpoints/global_step_5000/mp_rank_00_model_states.pt \
  --model.model_config=/mnt/bn/ecom-ccr-dev/mlx/users/doushihan/hf_models/bloomz-1b1 \
  --model.network.pad_idx=3 \
  --play-file-type="qa_batch" \
  --play-file-bsz 64 \
  --generate-steps 30 \
  --play-file /mnt/bn/ecom-ccr-dev/mlx/users/doushihan/data/junjun_v1/test_0130_add_trans2_prompt.parquet \
  --play-out-file /mnt/bn/ecom-ccr-dev/mlx/users/doushihan/playfile_output/test0406.jsonl \
  --generate-temp 0.7 \
  --generate-policy

bash launch.sh tasks/gpt2/model.py \
  --model.use_hf_ckpt=True \
  --data.from_hf_tokenizer=True \
  --data.tokenizer=/mnt/bn/ecom-ccr-dev/mlx/users/doushihan/hf_models/alpaca-native \
  --data.hf_tokenizer_use_fast=False \
  --data.max_seq_len=300 \
  --model=tasks/gpt2/1b.yaml \
  --model.partial_pretrain=/mnt/bn/ecom-ccr-dev/mlx/users/doushihan/hf_models/alpaca-native \
  --model.model_config=/mnt/bn/ecom-ccr-dev/mlx/users/doushihan/hf_models/alpaca-native \
  --model.network.pad_idx=1 \
  --play-file-type="qa_batch" \
  --play-file-bsz 64 \
  --generate-steps 30 \
  --play-file /mnt/bn/ecom-ccr-dev/mlx/users/doushihan/data/junjun_v1/GB_test_0130_add_trans2_prompt.parquet \
  --play-out-file /mnt/bn/ecom-ccr-dev/mlx/users/doushihan/playfile_output/yangtuo_GB_test_0130_add_trans2_prompt_output.jsonl \
  --generate-temp 0.7



bash launch.sh tasks/gpt2/model.py \
  --model.use_hf_ckpt=False \
  --data.from_hf_tokenizer=True \
  --data.tokenizer=/mnt/bn/ecom-ccr-dev/mlx/users/doushihan/hf_models/bloomz-560m \
  --data.hf_tokenizer_use_fast=False \
  --data.max_seq_len=300 \
  --model=tasks/gpt2/1b.yaml \
  --model.partial_pretrain=/mnt/bn/ecom-ccr-dev/mlx/users/doushihan/finetune_models/bloom/bloom560m-instruct-tuning-ccr71-lossmask-new-0419-debug-v4/checkpoints/global_step_2500/mp_rank_00_model_states.pt \
  --model.model_config=/mnt/bn/ecom-ccr-dev/mlx/users/doushihan/hf_models/bloomz-560m \
  --model.network.pad_idx=3 \
  --play-file-type="qa_batch" \
  --play-file-bsz 1 \
  --generate-steps 100 \
  --play-file /mnt/bn/ecom-ccr-dev/mlx/users/doushihan/data/ccr_jj_cls71/test_0130_add_trans2_prompt2.parquet \
  --play-out-file /mnt/bn/ecom-ccr-dev/mlx/users/doushihan/playfile_output/bloom560m-instruct-tuning-ccr71-epoch5-lossmask-output-debug.jsonl \
  --generate-temp 0.7