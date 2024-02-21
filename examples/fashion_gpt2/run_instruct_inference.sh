task_id=$(date +%Y%m%d%H)

bash launch.sh tasks/gpt2/unsup/model.py \
    --model=tasks/gpt2/unsup/1b3_v1.yaml \
    --play-file-type="qa" \
    --generate-temp=0.7 \
    --generate-trial-num=1 \
    --generate-steps=32 \
    --data.tokenizer=hdfs://tokenizer \
    --output_file_path=/mnt/bn/wanli/experiments/instruct_tunning/outputs/inference_result_${task_id}.txt \
    --play-file=/mnt/bn/wanli/experiments/instruct_tunning/data/ccr_mlc_valid.jsonl \
    --model.partial_pretrain=/mnt/bn/wanli/experiments/instruct_tunning/outputs/2023031416/checkpoints/global_step_34680/mp_rank_00_model_states.pt