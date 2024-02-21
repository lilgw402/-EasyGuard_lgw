# 基于量化+xperf加速之后的推理
# 关于量化+xperf的加速,可以参考文档:
# https://bytedance.feishu.cn/wiki/FVtfwQtbJiOlYQkHGs6c5Mwcn7f

import logging
import time
from typing import Dict, List, Union

import torch
import xperf_gpt
from quant import quant_model
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

xperf_gpt.load_xperf_gpt()


logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


class EndpointHandler:
    def __init__(self):
        # 使用skip可以加快模型构建
        def skip(*args, **kwargs):
            pass

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip
        self.model_path = "llama_7b_ccr/"
        self.hf_config = AutoConfig.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, max_len=1000000007)
        self.tokenizer.add_special_tokens(
            {"bos_token": "<s>", "unk_token": "<unk>", "eos_token": "</s>", "pad_token": "<pad>"}
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=self.hf_config,
            torch_dtype=torch.float16,  # 默认使用float16加载,减少CPU占用
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()
        quant_model(self.model)  # 模型量化
        self.xperf_generator = xperf_gpt.init_inference(  # 基于xperf优化
            self.model,
            max_batch_size=128,  # 实际forward的时候，输入的batch不能大于max_batch_size
            max_length=512,
            use_xperf_gpt=True,
            do_script=True,
        )
        # 对GPU做warmup，必不可少
        warmup_textx = {"text": ["I am a teacher".encode("UTF-8")]}
        self(warmup_textx)
        self(warmup_textx)
        self(warmup_textx)
        logger.info("Hi, there is a CCR ChatBot, You can ask me questions related CCR?")

    def __call__(
        self, request: Dict[str, Union[List[bytes], List[int], List[float]]]
    ) -> Dict[str, Union[List[bytes], List[int], List[float]]]:
        # 注意输入和返回值必须要满足这样的格式
        start = time.time()
        text = [t.decode(encoding="UTF-8") for t in request["text"]]
        token_out = self.tokenizer(text, padding=True, return_tensors="pt")
        lengths = token_out.input_ids.size(1)
        input_ids = token_out.input_ids.cuda()
        attention_mask = token_out.attention_mask.cuda()

        outputs = self.xperf_generator(
            input_ids,
            attention_mask=attention_mask,
        )
        generated = []
        for index in range(len(outputs)):
            output = outputs[index][lengths:]
            generated.append(self.tokenizer.decode(output, skip_special_tokens=True).encode(encoding="UTF-8"))
        logger.info(f"each infer time: {time.time() - start}")
        return dict(generated=generated)


if __name__ == "__main__":
    import json

    file = open("llama_7b_ccr/result_product_report_test_epoch5.jsonl", "r", encoding="utf-8")
    papers = []
    for line in file.readlines():
        dic = json.loads(line)
        papers.append(dic)
    logger.info(len(papers))
    bs = 128
    endpoint_handler = EndpointHandler()
    warmup_textx = {
        "text": ["I am a teacher".encode("UTF-8")],
    }
    endpoint_handler(warmup_textx)
    endpoint_handler(warmup_textx)
    endpoint_handler(warmup_textx)
    outputs = []
    spend_time = 0
    for i in range(0, len(papers), bs):
        if i % bs == 0:
            logger.info(i)
        texts = []
        for j in range(bs):
            if (i + j) < len(papers):
                texts.append(papers[i + j]["question"].encode(encoding="UTF-8"))
            else:
                break
        request = {"text": texts}
        start = time.time()
        output = endpoint_handler(request)
        spend_time += time.time() - start
        outputs.extend(output["generated"])
    logger.info(f"spend time: {spend_time}")

    with open("quant_xperf_bs256.json", "w", encoding="utf-8") as json_file:
        for index in tqdm(range(len(papers))):
            output = outputs[index]
            output = output.decode()

            papers[index].update({"vllm_answer": output})
            papers[index].pop("probs")
            papers[index].pop("question")
            json_str = json.dumps(papers[index], ensure_ascii=False)
            json_file.write(json_str)
            json_file.write("\n")
