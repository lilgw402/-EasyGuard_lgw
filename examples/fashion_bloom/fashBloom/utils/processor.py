import json


def ocnli_processor(line):
    data_dict = json.loads(line)
    if data_dict["label"] == "entailment":
        label_id = 1
    elif data_dict["label"] == "contradiction":
        label_id = 2
    else:
        label_id = 0
    data_dict["label_name"] = data_dict["label"]
    data_dict["label"] = label_id

    return data_dict


def rte_processor(line):
    data_dict = json.loads(line)
    if data_dict["label"] == "entailment":
        label_id = 0
    elif data_dict["label"] == "not_entailment":
        label_id = 1
    else:
        label_id = -1
    data_dict["label_name"] = data_dict["label"]
    data_dict["label"] = label_id

    return data_dict


def lambada_processor(line):
    data_dict = {"text": line, "label": None, "label_name": "null"}

    return data_dict
