declare -A model_config
declare -A model_ckpt
model_config=(
    ['aml_1b3_120k']='tasks/gpt2/zero_shot_eval/1b.yaml'
    ['aml_1b3_300k']='tasks/gpt2/zero_shot_eval/1b.yaml'
    ['aml_1b3_500k']='tasks/gpt2/zero_shot_eval/1b.yaml'
    ['aml_1b3_690k']='tasks/gpt2/zero_shot_eval/1b.yaml'
    ['aml_13b_78k']='tasks/gpt2/zero_shot_eval/13b_v1.yaml'
    ['aml_13b_99k']='tasks/gpt2/zero_shot_eval/13b_v1.yaml'
    ['alice_1.3b_292k']='tasks/gpt2/zero_shot_eval/alice1b3.yaml'
    ['alice_1.3b_400k']='tasks/gpt2/zero_shot_eval/alice1b3.yaml'
)
model_ckpt=(
    ['aml_1b3_120k']='hdfs://haruna/home/byte_data_aml_research/user/zhangzhi.joshua/models/gpt/1b_cleandata_v1_20220113/checkpoints/global_step_120000/mp_rank_00_model_states.pt'
    ['aml_1b3_300k']='hdfs://haruna/home/byte_data_aml_research/user/zhangzhi.joshua/models/gpt/1b_cleandata_v1_20220113/checkpoints/global_step_300000/mp_rank_00_model_states.pt'
    ['aml_1b3_500k']='hdfs://haruna/home/byte_data_aml_research/user/zhangzhi.joshua/models/gpt/1b_cleandata_v1_20220113/checkpoints/global_step_500000/mp_rank_00_model_states.pt'
    ['aml_1b3_690k']='hdfs://haruna/home/byte_data_aml_research/user/zhangzhi.joshua/models/gpt/1b_cleandata_v1_20220113/checkpoints/global_step_690000/mp_rank_00_model_states.pt'
    ['aml_13b_78k']='hdfs://haruna/home/byte_data_aml_research/user/zhangzhi.joshua/models/gpt/13b_cleandata_v1_20220113/checkpoints/global_step_78000/zero3_merge_states.pt'
    ['aml_13b_99k']='hdfs://haruna/home/byte_data_aml_research/user/zhangzhi.joshua/models/gpt/13b_cleandata_v1_20220113/checkpoints/global_step_99000/zero3_merge_states.pt'
    ['alice_1.3b_292k']='hdfs://haruna/home/byte_search_nlp_cr/user/huangwenguan/model/alice/gpt1b3_dpsp_bsz16x8x8x1_400/model_state_epoch_292000.th'
    ['alice_1.3b_400k']='hdfs://haruna/home/byte_search_nlp_cr/user/huangwenguan/model/alice/gpt1b3_dpsp_bsz16x8x8x1_400/model_state_epoch_400000.th'
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
    # printf "${model_config[$@]}\t${model_ckpt[$@]}\t${dataset_name_array[$i]}\t${subset_name_array[$i]}\t${template_name_array[$i]}\n"
    bash launch.sh tasks/gpt2/zero_shot_eval/model.py --model="${model_config[$@]}" --model.partial_pretrain="${model_ckpt[$@]}" --data.val_num_workers=1 --data.val_batch_size=1 --trainer.val_check_interval=1.0 --data.val_path="${val_path_array[$i]}" --data.dataset_name="${dataset_name_array[$i]}" --data.subset_name="${subset_name_array[$i]}" --data.template_name="${template_name_array[$i]}" --val-only --model.network.use_rmpad_lmloss=false --model.network.use_rmpad_lnmlp=false --model.network.use_rmpad_attn=false --model.network.pad_idx=2 --data.max_seq_len=512
done

