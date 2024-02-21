import argparse
import torch
from torch.utils.data import DataLoader
from transformers import LlamaTokenizer
from valley.model.language_model.valley_llama import ValleyVideoLlamaForCausalLM, ValleyProductLlamaForCausalLM
import torch
import os
import sys
import json
from valley.utils import disable_torch_init
import os
import random
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
import traceback
from torch.utils.data.distributed import DistributedSampler
from valley.util.config import DEFAULT_GANDALF_TOKEN
from valley.util.data_util import KeywordsStoppingCriteria
from peft import PeftConfig, PeftModel
from transformers import set_seed
from valley.data.dataset import LazySupervisedDataset, DataCollatorForSupervisedDataset
from valley.util.data_util import smart_tokenizer_and_embedding_resize
from valley import conversation as conversation_lib
from valley.util.config import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, DEFAULT_VIDEO_FRAME_TOKEN, DEFAULT_VI_START_TOKEN, DEFAULT_VI_END_TOKEN


os.environ['NCCL_DEBUG']=''
def setup(args,rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.DDP_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size, )

def standardization(data):
        mu = torch.mean(data)
        sigma = torch.std(data)
        return (data - mu) / sigma

def inference(rank, world_size, args):
    set_seed(42)

    this_rank_gpu_index = rank

    if args.DDP:
        torch.cuda.set_device(this_rank_gpu_index)
        setup(args, rank, world_size)
        
    disable_torch_init()

    device = torch.device('cuda:'+str(this_rank_gpu_index)
                          if torch.cuda.is_available() else 'cpu')

    Model = None
    if args.model_class == 'valley-video':
        Model = ValleyVideoLlamaForCausalLM
    elif args.model_class == 'valley-product':
        Model = ValleyProductLlamaForCausalLM

    model_name = os.path.expanduser(args.model_name)

    # load model
    if 'lora' in model_name:
        print('load model')
        # model_old = Model.from_pretrained(model_name, torch_dtype=torch.float16)

        config = PeftConfig.from_pretrained(model_name)
        model_old = Model.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.float16)
        model_old = PeftModel.from_pretrained(model_old, model_name)
        model_old = model_old.merge_and_unload().half()
        print('load lora end')
        
        if os.path.exists(os.path.join(model_name,'non_lora_trainables.bin')):
            non_lora_state_dict = torch.load(os.path.join(model_name,'non_lora_trainables.bin'))
            new_state_dict = dict()
            for key in non_lora_state_dict.keys():
                key_new = '.'.join(key.split('.')[2:]) # base_model.model.model.xxxx
                new_state_dict[key_new] = non_lora_state_dict[key]
            model_old_state = model_old.state_dict()
            model_old_state.update(new_state_dict)
            model_old.load_state_dict(model_old_state)
        model = model_old
        tokenizer = LlamaTokenizer.from_pretrained(config.base_model_name_or_path, use_fast = False)
        tokenizer.padding_side = 'left'
        print("load end")
    else:
        print('load model')
        model = Model.from_pretrained(
            model_name, torch_dtype=torch.float16)
        tokenizer = LlamaTokenizer.from_pretrained(model_name, use_fast = False)
        tokenizer.padding_side = 'left'
        print('load end')
    
    if args.language == 'chinese':
        from transformers import ChineseCLIPImageProcessor as CLIPImageProcessor
    else:
        from transformers import  CLIPImageProcessor
    
    image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower)
    model.eval()
    model = model.to(device)

    args.image_processor = image_processor
    args.is_multimodal = True
    args.mm_use_im_start_end = True
    args.only_mask_system = False

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    vision_tower = model.get_model().vision_tower   
    vision_tower.to(device, dtype=torch.float16)
    vision_config = vision_tower.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
        vision_config.vi_start_token, vision_config.vi_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_VI_START_TOKEN, DEFAULT_VI_END_TOKEN])
        vision_config.vi_frame_token = tokenizer.convert_tokens_to_ids(DEFAULT_VIDEO_FRAME_TOKEN)
    
    if args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if args.prompt_version is not None:
        conversation_lib.default_conversation = conversation_lib.conv_templates[args.prompt_version]
    dataset = LazySupervisedDataset(args.data_path, tokenizer=tokenizer, data_args = args, inference= True)

    if args.DDP:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=DataCollatorForSupervisedDataset(tokenizer=tokenizer, inference=True), pin_memory=True, sampler=sampler,)
        rf = open(args.out_path+".worker_"+str(rank), 'w')
    else:
        dataloader = DataLoader(dataset, num_workers=1, batch_size=args.batch_size, collate_fn=DataCollatorForSupervisedDataset(tokenizer=tokenizer, inference=True), pin_memory=True)
        rf = open(args.out_path, 'w')

    prog_bar = tqdm(dataloader, total=len(dataloader),desc='worker_'+str(rank)) if rank == 0 else dataloader

    for test_batch in prog_bar:
        data_id = test_batch.pop('id')
        input_ids = test_batch['input_ids'].to(device)
        attention_mask = test_batch['attention_mask'].to(device)
        images = [img.half().to(device) for img in test_batch['images']]
        stop_str = conversation_lib.default_conversation.sep if conversation_lib.default_conversation.sep_style != conversation_lib.SeparatorStyle.TWO else conversation_lib.default_conversation.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids = input_ids,
                images = images,
                attention_mask = attention_mask,
                do_sample=args.do_sample,
                temperature=args.temperature,
                stopping_criteria=[stopping_criteria],
                max_new_tokens = 5,
                # pad_token_id = tokenizer.pad_token_id,
                return_dict_in_generate= True if args.ouput_logits else False, 
                output_scores= True if args.ouput_logits else False
            )

        if not args.ouput_logits:
            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode( output_ids[:, input_token_len:], skip_special_tokens=True)
            response = process_response(outputs)
            print(response)
        else:
            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids.sequences[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(
                    f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids.sequences[:, input_token_len:], skip_special_tokens=True)
            response = process_response(outputs)

            scores = standardization(output_ids.scores[3]) # 3 代表 输出 yes 和 no 的那一个位置的 idx
            standardization_score = scores[:,[34043,29871]] # scores[:,[34043,31191]]
            standardization_logits = torch.softmax(standardization_score.to(torch.float16), dim=1).cpu().numpy().tolist()
            generated_scores = [format(yes_logits, '.6f') for yes_logits, _ in standardization_logits]

            print(response)
            print(generated_scores)

        for i in range(len(response)):
            rf.write(data_id[i] + "\t" + response[i].replace('\n','') + '\n')
            rf.flush()
    rf.close()


def process_response(outputs):
        output = []
        for i, out in enumerate(outputs):
            while True:
                cur_len = len(out)
                out = out.strip()
                for pattern in ['###', 'Assistant:', 'Response:', 'Valley:']:
                    if out.startswith(pattern):
                        out = out[len(pattern):].strip()
                if len(out) == cur_len:
                    break
            try:
                index = out.index('###')
            except ValueError:
                out += '###'
                index = out.index("###")
            out = out[:index].strip()
            output.append(out)
        return output


def gather_result(args,world_size):
    num_worker = world_size
    with open(args.out_path, 'w') as f:
        for i in range(num_worker):
            with open(args.out_path+".worker_"+str(i), 'r') as tf:
                tmp_result = tf.readlines()
            f.writelines(tmp_result)
            os.remove(args.out_path+".worker_"+str(i))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-class", type=str, default="valley-product")
    parser.add_argument("--language", type=str, default="chinese")
    parser.add_argument("--model-name", type=str, default = '/mnt/bn/yangmin-priv-fashionmm/Data/wuji/data_process/new_process/product_checkpoints/data-ys-v4-valley-product-7b-jinshou-class-lora-multi-class/checkpoint-50000')
    parser.add_argument("--video_data_path", type=str, required = False, default = None)
    parser.add_argument("--data_path", type=str, required = False, default = '/mnt/bn/yangmin-priv-fashionmm/Data/wuji/wupian_process/new_wupian/wupian_test_data_4w_ocr512_n_valley_product.json')
    parser.add_argument("--video_folder", type=str, required = False, default = None)
    parser.add_argument("--image_folder", type=str, required = False, default = '/mnt/bn/yangmin-priv-fashionmm/Data/wuji/big_model_wupian_image_data')
    parser.add_argument("--out_path", type=str, required = False, default = '/mnt/bn/yangmin-priv-fashionmm/Checkpoints/test_output_debug.txt' )
    parser.add_argument("--version", type=str, default="v0")
    parser.add_argument("--prompt_version", type=str, default="jinshou_cot")
    parser.add_argument("--max_img_num", type=int, default=12)
    parser.add_argument("--image_aspect_ratio", type=str, default=None)
    parser.add_argument("--batch_size", type=int, required=False, default=2)
    parser.add_argument("--ouput_logits", action="store_true", default=True)
    parser.add_argument("--temperature", type = float, default=1)
    parser.add_argument("--do-sample", action="store_true", default=False)
    parser.add_argument("--DDP", action="store_true")
    parser.add_argument("--DDP_port", default = '12345')
    parser.add_argument("--world_size", type=int, default = 8)
    args = parser.parse_args()

    if args.DDP:
        mp.spawn( inference, args=(args.world_size, args), nprocs=args.world_size)
        gather_result(args, args.world_size)
    else: 
        inference(0, args.world_size, args)