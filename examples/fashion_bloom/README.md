# some config
```bash
# launch GPUs
launch --gpu 1 --cpu 20 --memory 300 --type a100-80g -- doas --krb5-username doushihan bash
```
```bash
# my hdfs
hdfs://haruna/home/byte_ecom_govern/user/doushihan
hdfs://haruna/home/byte_ecom_govern/user/doushihan/models/gpt2/1b3_0221_mini_data/
/mnt/bn/ecom-govern-maxiangqian/doushihan/play_file_out/outputs/easyguard_play_file_sample_output.jsonl
hdfs://harunava/home/byte_magellan_va/user/
hdfs://harunava/home/byte_magellan_va/user/doushihan/models/bloom7b1-chatcat-bsz4-ga4-wp-60k-v2-0407/checkpoints/global_step_1207/mp_rank_00_model_states.pt

cd /opt/tiger/EasyGuard/examples/fashion_bloom/
```

```bash
# image config
export BYTED_TORCH_AUTO_UPDATE=off
export BYTED_TORCH_BYTECCL=O0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
```
