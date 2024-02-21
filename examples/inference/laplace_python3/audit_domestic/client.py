import time

import datasets

# pip3 install bytedlaplace
from laplace import Client

if __name__ == "__main__":
    inputs_path = "chinese-alpaca-2-13b-1008/audit_LLM_benchmark_0810_clean_v2_new_field_qa.jsonl"
    test_dataset = datasets.load_dataset("json", data_files={"test": inputs_path})["test"]
    # ecom.govern.chinese_alpaca_2_13b_8bit替换成自己的PSM
    # hl替换成自己的集群
    laplace_client = Client("sd://ecom.govern.chinese_alpaca_2_13b_8bit?idc=hl&cluster=default", timeout=1)
    for i in range(len(test_dataset)):
        data_list = []
        data_list.append(test_dataset[i]["instruction"].encode())
        request = {"text": data_list}
        start = time.time()
        # bernard_chinese_alpaca_2_13b_8bit替换成自己的模型名称
        results = laplace_client.matx_inference("bernard_chinese_alpaca_2_13b_8bit", request)
        results = results.output_bytes_lists["generate"]
        print("each infer time: ", time.time() - start)
        outputs = []
        for output_text in results:
            output_text = output_text.decode()
            outputs.append(output_text)
        print("label: ", test_dataset[i]["output"])
        print("predict: ", output_text)
