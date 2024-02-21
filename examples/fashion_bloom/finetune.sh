# cd /mnt/bn/ecom-ccr-dev/mlx/users/doushihan/EasyGuard/examples/fashion_bloom/ && pwd && bash launch.sh tasks/gpt2/model.py \
#   --model.use_hf_ckpt=False \
#   --data.from_hf_tokenizer=True \
#   --data.tokenizer=/mnt/bn/ecom-ccr-dev-878d5f0f/mlx/users/doushihan/hf_models/bloomz-1b1 \
#   --data.hf_tokenizer_use_fast=False \
#   --data.max_seq_len=512 \
#   --model=tasks/gpt2/1b.yaml \
#   --model.partial_pretrain=/mnt/bn/ecom-ccr-dev-878d5f0f/mlx/users/doushihan/finetune_models/bloom/bloo1b1_0316_100B_round1/checkpoints/global_step_600000/mp_rank_00_model_states.pt \
#   --model.model_config=/mnt/bn/ecom-ccr-dev-878d5f0f/mlx/users/doushihan/hf_models/bloomz-1b1 \
#   --data.train_num_workers=1 \
#   --data.train_batch_size=16 \
#   --trainer.accumulate_grad_batches=2 \
#   --data.val_num_workers=1 \
#   --data.val_batch_size=16 \
#   --trainer.val_check_interval=1 \
#   --data.train_path=/mnt/bn/ecom-ccr-dev-878d5f0f/mlx/users/doushihan/data/junjun_v1/train_0130_add_trans2_prompt.parquet \
#   --data.val_path=/mnt/bn/ecom-ccr-dev-878d5f0f/mlx/users/doushihan/data/junjun_v1/test_0130_add_trans2_prompt.parquet \
#   --data.train_size=-1 \
#   --data.dyn_bsz=False \
#   --data.dyn_bsz_margin=0 \
#   --data.stride=1920 \
#   --data.bsz_warmup=False \
#   --data.finetune_type_is_qa=True \
#   --data.drop_last=True \
#   --model.network.use_rmpad_lmloss=false \
#   --model.network.use_rmpad_lnmlp=false \
#   --model.network.use_rmpad_attn=false \
#   --model.network.pad_idx=3 \
#   --trainer.max_epochs=4 \
#   --trainer.optimizer_kwargs.optimizer.params.lr=1e-5 \
#   --trainer.default_hdfs_dir=/mnt/bn/ecom-ccr-dev-878d5f0f/mlx/users/doushihan/finetune_models/bloom/bloom1b1-cp-ft-ccr-3cls \
#   --trainer=tasks/gpt2/zero2-bf16.yaml


cd /mnt/bn/ecom-ccr-dev-878d5f0f/mlx/users/doushihan/EasyGuard/examples/fashion_bloom/ && pwd && bash launch.sh tasks/gpt2/model.py \
  --model.use_hf_ckpt=True \
  --data.from_hf_tokenizer=True \
  --data.tokenizer=/mnt/bn/ecom-ccr-dev-878d5f0f/mlx/users/doushihan/hf_models/bloomz-1b1 \
  --data.hf_tokenizer_use_fast=False \
  --data.max_seq_len=512 \
  --model=tasks/gpt2/1b.yaml \
  --model.partial_pretrain=/mnt/bn/ecom-ccr-dev-878d5f0f/mlx/users/doushihan/hf_models/bloomz-1b1 \
  --model.model_config=/mnt/bn/ecom-ccr-dev-878d5f0f/mlx/users/doushihan/hf_models/bloomz-1b1 \
  --data.train_num_workers=1 \
  --data.train_batch_size=16 \
  --trainer.accumulate_grad_batches=2 \
  --data.val_num_workers=1 \
  --data.val_batch_size=16 \
  --trainer.val_check_interval=1 \
  --data.train_path=/mnt/bn/ecom-ccr-dev-878d5f0f/mlx/users/doushihan/data/ccr_instruct_tuning \
  --data.val_path=/mnt/bn/ecom-ccr-dev-878d5f0f/mlx/users/doushihan/data/ccr_jj_cls71/test_0130_add_trans2_prompt2.parquet \
  --data.train_size=-1 \
  --data.dyn_bsz=False \
  --data.dyn_bsz_margin=0 \
  --data.stride=1920 \
  --data.bsz_warmup=False \
  --data.finetune_type_is_qa=True \
  --data.use_loss_mask=False \
  --data.drop_last=True \
  --model.network.use_rmpad_lmloss=false \
  --model.network.use_rmpad_lnmlp=false \
  --model.network.use_rmpad_attn=false \
  --model.network.pad_idx=3 \
  --trainer.max_epochs=4 \
  --trainer.optimizer_kwargs.optimizer.params.lr=1e-5 \
  --trainer.default_hdfs_dir=/mnt/bn/ecom-ccr-dev-878d5f0f/mlx/users/doushihan/finetune_models/bloom/bloom1b1-instruct-tuning-ccr71-0324 \
  --trainer=tasks/gpt2/zero2-bf16-warmup.yaml






  # 0330 my alpaca
    cd /mnt/bn/ecom-ccr-dev-878d5f0f/mlx/users/doushihan/EasyGuard/examples/fashion_bloom/ && pwd && bash launch.sh tasks/gpt2/model.py \
  --model.use_hf_ckpt=True \
  --data.from_hf_tokenizer=True \
  --data.tokenizer=/mnt/bn/ecom-ccr-dev-878d5f0f/mlx/users/doushihan/hf_models/bloomz-7b1 \
  --data.hf_tokenizer_use_fast=False \
  --data.max_seq_len=2048 \
  --model=tasks/gpt2/1b.yaml \
  --model.partial_pretrain=/mnt/bn/ecom-ccr-dev-878d5f0f/mlx/users/doushihan/hf_models/bloomz-7b1 \
  --model.model_config=/mnt/bn/ecom-ccr-dev-878d5f0f/mlx/users/doushihan/hf_models/bloomz-7b1 \
  --data.train_num_workers=1 \
  --data.train_batch_size=4 \
  --trainer.accumulate_grad_batches=1 \
  --data.val_num_workers=1 \
  --data.val_batch_size=4 \
  --trainer.val_check_interval=1 \
  --data.train_path=/mnt/bn/ecom-ccr-dev-878d5f0f/mlx/users/doushihan/data/dialog_60k/clean.parquet \
  --data.val_path=/mnt/bn/ecom-ccr-dev-878d5f0f/mlx/users/doushihan/data/dialog_60k/clean.parquet \
  --data.train_size=26549561 \
  --data.dyn_bsz=True \
  --data.dyn_bsz_margin=0 \
  --data.stride=1920 \
  --data.bsz_warmup=True \
  --data.drop_last=False \
  --data.finetune_type_is_qa=False \
  --data.use_loss_mask=False \
  --model.network.use_rmpad_lmloss=false \
  --model.network.use_rmpad_lnmlp=false \
  --model.network.use_rmpad_attn=false \
  --model.network.pad_idx=3 \
  --trainer.max_epochs=5 \
  --trainer.optimizer_kwargs.optimizer.params.lr=1e-5 \
  --trainer.default_hdfs_dir=/mnt/bn/ecom-ccr-dev-878d5f0f/mlx/users/doushihan/finetune_models/bloom/bloom7b1-chatcat-bsz4-warmup-1n8g \
  --trainer=tasks/gpt2/zero2-bf16-warmup.yaml




  cd /mnt/bn/ecom-ccr-dev/mlx/users/doushihan/EasyGuard/examples/fashion_bloom/ && pwd && bash launch.sh tasks/gpt2/model.py \
  --model.use_hf_ckpt=True \
  --data.from_hf_tokenizer=True \
  --data.tokenizer=/mnt/bn/ecom-ccr-dev/mlx/users/doushihan/hf_models/bloomz-3b \
  --data.hf_tokenizer_use_fast=False \
  --data.max_seq_len=512 \
  --model=tasks/gpt2/1b.yaml \
  --model.partial_pretrain=/mnt/bn/ecom-ccr-dev/mlx/users/doushihan/hf_models/bloomz-3b \
  --model.model_config=/mnt/bn/ecom-ccr-dev/mlx/users/doushihan/hf_models/bloomz-3b \
  --data.train_num_workers=1 \
  --data.train_batch_size=1 \
  --trainer.accumulate_grad_batches=4 \
  --data.val_num_workers=1 \
  --data.val_batch_size=1 \
  --trainer.val_check_interval=0.5 \
  --data.train_path=/mnt/bn/ecom-ccr-dev/mlx/users/doushihan/data/dialog_60k/trainset_v2_0407_60kwithwk \
  --data.val_path=/mnt/bn/ecom-ccr-dev/mlx/users/doushihan/data/dialog_60k/trainset_v2_0407_60kwithwk \
  --data.train_size=26549561 \
  --data.dyn_bsz=True \
  --data.dyn_bsz_margin=0 \
  --data.stride=1920 \
  --data.bsz_warmup=True \
  --data.drop_last=False \
  --data.finetune_type_is_qa=False \
  --data.use_loss_mask=False \
  --model.network.use_rmpad_lmloss=false \
  --model.network.use_rmpad_lnmlp=false \
  --model.network.use_rmpad_attn=false \
  --model.network.pad_idx=3 \
  --trainer.max_epochs=5 \
  --trainer.optimizer_kwargs.optimizer.params.lr=1e-5 \
  --trainer.default_hdfs_dir=/mnt/bn/ecom-ccr-dev/mlx/users/doushihan/finetune_models/bloom/bloom3b-1gpu-test \
  --trainer=tasks/gpt2/zero2-bf16-warmup.yaml





bash launch.sh tasks/gpt2/model.py \
  --model.use_hf_ckpt=True \
  --data.from_hf_tokenizer=True \
  --data.tokenizer=/mnt/bn/ecom-ccr-dev/mlx/users/doushihan/hf_models/bloomz-560m \
  --data.hf_tokenizer_use_fast=False \
  --data.max_seq_len=512 \
  --model=tasks/gpt2/1b.yaml \
  --model.partial_pretrain=/mnt/bn/ecom-ccr-dev/mlx/users/doushihan/hf_models/bloomz-560m \
  --model.model_config=/mnt/bn/ecom-ccr-dev/mlx/users/doushihan/hf_models/bloomz-560m \
  --data.train_num_workers=1 \
  --data.train_batch_size=2 \
  --trainer.accumulate_grad_batches=3 \
  --data.val_num_workers=1 \
  --data.val_batch_size=2 \
  --trainer.val_check_interval=1 \
  --data.train_path=/mnt/bn/ecom-ccr-dev/mlx/users/doushihan/data/ccr_it_v2 \
  --data.val_path=hdfs://harunava/home/byte_magellan_va/user/jiangjunjun.happy/ccr_v3/prompt_data/test_0130_add_trans2_prompt2.parquet \
  --data.train_size=-1 \
  --data.dyn_bsz=False \
  --data.dyn_bsz_margin=0 \
  --data.stride=1920 \
  --data.bsz_warmup=False \
  --data.finetune_type_is_qa=True \
  --data.drop_last=True \
  --model.network.use_rmpad_lmloss=false \
  --model.network.use_rmpad_lnmlp=false \
  --model.network.use_rmpad_attn=false \
  --model.network.pad_idx=3 \
  --data.use_loss_mask=True \
  --trainer.max_epochs=5 \
  --trainer.optimizer_kwargs.optimizer.params.lr=1e-5 \
  --trainer.default_hdfs_dir=/mnt/bn/ecom-ccr-dev/mlx/users/doushihan/finetune_models/bloom/lossmask-debug \
  --trainer=tasks/gpt2/zero2-bf16-warmup.yaml





  ## loss mask debug 0419
  bash launch.sh tasks/gpt2/model.py \
  --model.use_hf_ckpt=True \
  --data.from_hf_tokenizer=True \
  --data.tokenizer=/mnt/bn/ecom-ccr-dev/mlx/users/doushihan/hf_models/bloomz-560m \
  --data.hf_tokenizer_use_fast=False \
  --data.max_seq_len=512 \
  --model=tasks/gpt2/1b.yaml \
  --model.partial_pretrain=/mnt/bn/ecom-ccr-dev/mlx/users/doushihan/hf_models/bloomz-560m \
  --model.model_config=/mnt/bn/ecom-ccr-dev/mlx/users/doushihan/hf_models/bloomz-560m \
  --data.train_num_workers=1 \
  --data.train_batch_size=8 \
  --trainer.accumulate_grad_batches=2 \
  --data.val_num_workers=1 \
  --data.val_batch_size=8 \
  --trainer.val_check_interval=0.05 \
  --data.train_path=hdfs://harunava/home/byte_magellan_va/user/jiangjunjun.happy/ccr_v3/prompt_data/test_0130_add_trans2_prompt2.parquet \
  --data.val_path=hdfs://harunava/home/byte_magellan_va/user/jiangjunjun.happy/ccr_v3/prompt_data/test_0130_add_trans2_prompt2.parquet \
  --data.train_size=-1 \
  --data.dyn_bsz=False \
  --data.dyn_bsz_margin=0 \
  --data.stride=1920 \
  --data.bsz_warmup=False \
  --data.finetune_type_is_qa=True \
  --data.drop_last=True \
  --model.network.use_rmpad_lmloss=false \
  --model.network.use_rmpad_lnmlp=false \
  --model.network.use_rmpad_attn=false \
  --model.network.pad_idx=3 \
  --data.use_loss_mask=False \
  --trainer.max_epochs=5 \
  --trainer.optimizer_kwargs.optimizer.params.lr=1e-5 \
  --trainer.default_hdfs_dir=/mnt/bn/ecom-ccr-dev/mlx/users/doushihan/finetune_models/bloom/bloom560m-valloss-debug \
  --trainer=tasks/gpt2/zero2-bf16-warmup.yaml