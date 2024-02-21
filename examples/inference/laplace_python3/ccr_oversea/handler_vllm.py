# 基于vLLM加速之后的推理
# 关于vllm的使用，可以看官方使用文档:
# https://vllm.readthedocs.io/en/latest/getting_started/quickstart.html
# 在batch size很大的时候，可以极大的提升模型的吞吐量

import logging
import time
from typing import Dict, List, Union

from tqdm import tqdm
from vllm import LLM, SamplingParams

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


class EndpointHandler:
    def __init__(self):
        self.model_path = "llama_7b_ccr/"
        self.sampling_params = SamplingParams(
            max_tokens=512,
        )
        self.llm = LLM(model=self.model_path)
        warmup_textx = {"text": ["I am a teacher".encode("UTF-8")]}
        self(warmup_textx)
        self(warmup_textx)
        self(warmup_textx)
        logger.info("Hi, there is a CCR ChatBot, You can ask me questions related CCR?")

    def __call__(
        self, request: Dict[str, Union[List[bytes], List[int], List[float]]]
    ) -> Dict[str, Union[List[bytes], List[int], List[float]]]:
        text = [t.decode(encoding="UTF-8") for t in request["text"]]
        start = time.time()
        outputs = self.llm.generate(text, self.sampling_params)
        logger.info(f"each infer time: {time.time() - start}")
        for index in range(len(outputs)):
            output = outputs[index]
            generated_text = output.outputs[0].text
            outputs[index] = generated_text
        generated = [generated_text.encode() for generated_text in outputs]
        return dict(generated=generated)


if __name__ == "__main__":
    import json

    file = open("llama_7b_ccr/result_product_report_test_epoch5.jsonl", "r", encoding="utf-8")
    papers = []
    for line in file.readlines():
        dic = json.loads(line)
        papers.append(dic)
    logger.info(len(papers))
    texts = []
    for item in papers:
        texts.append(item["question"].encode(encoding="UTF-8"))
    endpoint_handler = EndpointHandler()
    start = time.time()
    length = len(texts)
    chunck_num = 1
    batch_size = length // chunck_num
    outputs = []
    for i in range(chunck_num + 1):
        texts_chunck = texts[i * batch_size : (i + 1) * batch_size]
        request = {"text": texts_chunck}
        output = endpoint_handler(request)
        outputs.extend(output["generated"])
    logger.info(f"spend time: {time.time() - start}")

    with open("vllm.json", "w", encoding="utf-8") as json_file:
        for index in tqdm(range(len(papers))):
            output = outputs[index]
            output = output.decode()
            papers[index].update({"vllm_answer": output})
            papers[index].pop("probs")
            papers[index].pop("question")
            json_str = json.dumps(papers[index], ensure_ascii=False)
            json_file.write(json_str)
            json_file.write("\n")
