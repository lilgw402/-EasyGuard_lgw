import json
llava_benchmark = [json.loads(line) for line in open('/mnt/bn/yangmin-priv-fashionmm/projects/zhaoziwang/data/chinese_valley_test_image/qa90_questions_zh.jsonl').readlines()]
llava_benchmark_answer = [json.loads(line) for line in open('/mnt/bn/yangmin-priv-fashionmm/projects/zhaoziwang/data/chinese_valley_test_image/qa90_gpt4_answer_zh.jsonl').readlines()]
image_path = '/mnt/bn/yangmin-priv-fashionmm/projects/zhaoziwang/data/chinese_valley_test_image/image/'


llava_bench_chat = []
for i,data in enumerate(llava_benchmark):
    assert llava_benchmark_answer[i]['question_id'] == data['question_id']
    this_dict = dict(
        id = data['question_id'],
        image = image_path+'/COCO_val2014_'+data['image'],
        conversations = [
            {'from':'human','value':data['text']+'\n<image>'}
        ],
        type = data['category'],
        gt_label = llava_benchmark_answer[i]['text'].replace('\n','\\n')
    )
    llava_bench_chat.append(this_dict)

json.dump(llava_bench_chat, open('llava_bench_chat.json','w'))