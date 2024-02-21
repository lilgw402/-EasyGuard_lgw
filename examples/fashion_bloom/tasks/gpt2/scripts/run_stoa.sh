declare -A model_dir
model_dir=(
    ['bloom_3b']='hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/hf_models/bloom-3b'
    ['bloom_7b1']='hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/hf_models/bloom-7b1'
    ['opt_1b3']='hdfs://haruna/home/byte_data_aml_research/user/guokun.lai/hf_models/opt-1.3b'
    ['opt_13b']='hdfs://haruna/home/byte_data_aml_research/user/guokun.lai/hf_models/opt-13b'
)

val_path_array=(
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/lambada/clean_test
    hdfs://haruna/home/byte_data_aml_research/user/taiqing.wang/corpus/lambda_zh_parquet
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/hellaswag/clean_dev
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/ai2_arc/ARC-Challenge/clean_test
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/ai2_arc/ARC-Easy/clean_test
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/super_glue/rte/clean_dev
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/clue/ocnli/clean_dev
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/super_glue/wsc/clean_dev
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/clue/chid/clean_dev
)

dataset_name_array=(
    lambada
    lambada_zh
    hellaswag
    ai2_arc
    ai2_arc
    super_glue
    clue
    super_glue
    clue
)

subset_name_array=(
    ""
    ""
    ""
    ARC-Challenge
    ARC-Easy
    rte
    ocnli
    wsc.fixed
    chid
)

template_name_array=(
    please+next+word
    please_next_word
    Open-ended+completion
    heres_a_problem
    pick_the_most_correct_option
    MNLI+crowdsource
    OCNLI+crowdsource
    does+the+pronoun+refer+to
    fill_the_blank
)

for i in "${!val_path_array[@]}"
do
    # printf "${model_dir[$@]}\t${model_dir[$@]}/config.json\t${dataset_name_array[$i]}\t${subset_name_array[$i]}\t${template_name_array[$i]}\n"
    bash launch.sh tasks/gpt2/zero_shot_eval/model.py --model.use_hf_ckpt=True --data.from_hf_tokenizer=True --data.tokenizer="${model_dir[$@]}" --data.hf_tokenizer_use_fast=False --data.max_seq_len=512 --model.partial_pretrain="${model_dir[$@]}" --model.model_config="${model_dir[$@]}/config.json" --data.val_num_workers=1 --data.val_batch_size=4 --trainer.val_check_interval=1.0 --data.val_path="${val_path_array[$i]}" --data.dataset_name="${dataset_name_array[$i]}" --data.subset_name="${subset_name_array[$i]}" --data.template_name="${template_name_array[$i]}"  --val-only
done