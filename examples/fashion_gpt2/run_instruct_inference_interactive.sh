python3 tasks/gpt2/unsup/model.py \
      --play \
      --generate-temp=0.7 \
      --model=tasks/gpt2/unsup/1b3_v1.yaml \
      --generate-steps=32 \
      --generate-trial-num=1 \
      --model.partial_pretrain=/mnt/bn/wanli/experiments/instruct_tunning/outputs/2023031416/checkpoints/global_step_34680/mp_rank_00_model_states.pt \
      --data.tokenizer=hdfs://tokenizer