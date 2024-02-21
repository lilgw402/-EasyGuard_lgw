import argparse

import datasets
from handler_quant_xperf import EndpointHandler

parser = argparse.ArgumentParser()

parser.add_argument(
    "--inputs_path",
    default="chinese-alpaca-2-13b-1008/audit_LLM_benchmark_0810_clean_v2_new_field_qa.jsonl",
    type=str,
)
parser.add_argument(
    "--save_path",
    default="quant8bit_xperf_audit_bs1_max_length1024_1008.csv",
    type=str,
)
args = parser.parse_args()


if __name__ == "__main__":
    inputs_path = args.inputs_path
    test_dataset = datasets.load_dataset("json", data_files={"test": inputs_path})["test"]
    print(len(test_dataset))
    endpoint_handler = EndpointHandler()
    batch_size = 1
    outputs = []
    for i in range(0, len(test_dataset), batch_size):
        if i % 1000 == 0:
            print(i)
        texts = []
        for j in range(batch_size):
            if (i + j) < len(test_dataset):
                texts.append(test_dataset[i + j])
            else:
                break
        request = {"text": [text["instruction"].encode() for text in texts]}
        generated = endpoint_handler(request)
        for i in range(len(generated["generate"])):
            output = generated["generate"][i].decode()
            print(output)
            outputs.append(output)
    test_dataset = test_dataset.add_column("model_predictions", outputs)
    test_dataset.to_csv(args.save_path)
