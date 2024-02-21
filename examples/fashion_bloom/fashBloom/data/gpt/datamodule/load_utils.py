import datasets
from torch.utils.data import Dataset


class local_collator:
    """
    Collates transformer outputs.
    """

    def __init__(self, pad_token_id=3):
        self._pad_token_id = pad_token_id

    def __call__(self, features):
        # Separate the list of inputs, outputs and labels
        # print(features)
        features_dict = {}
        for feat in features:
            for k, v in feat.items():
                if k in features_dict:
                    features_dict[k].append(v)
                else:
                    features_dict[k] = [v]

        model_inputs = features_dict["model_inputs"]

        padded_inputs = {}
        seq_len = [len(i) for i in [x["input_ids"] for x in model_inputs]]
        max_len = max(seq_len)

        for key in list(model_inputs[0].keys()):
            if key == "input_ids":
                padding_value = self._pad_token_id
            else:
                padding_value = 0

            sequence = [x[key] for x in model_inputs]

            for i in range(len(sequence)):
                sequence[i] = [padding_value] * (max_len - len(sequence[i])) + sequence[i]
                # sequence[i] = [padding_value]*20+sequence[i]

            padded_inputs[key] = sequence

        features_dict["padded_inputs"] = padded_inputs
        features_dict["seq_len"] = seq_len

        # print(features_dict)

        return features_dict


class local_dataset(Dataset):
    def __init__(
        self,
        hname,
        tokenizer,
        data_type="csv",
        shuffle=False,
    ):
        self.tokenizer = tokenizer
        self.type = "parquet"
        self.dataset = datasets.load_dataset(self.type, data_files={"test": hname})["test"]
        if shuffle:
            self.dataset.shuffle()

    def _format_examples(self, examples):
        ex_dict = {}
        keys = examples.keys()
        for k in keys:
            ex_dict[k] = examples[k]
        ex_dict["model_inputs"] = self.tokenizer(examples["question"].strip())

        # print('tokens: ', self.tokenizer.tokenize(examples['question'].strip()))

        return ex_dict

    def shuffle(self):
        self.dataset.shuffle()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        """Return i-th sample."""
        return self._format_examples(self.dataset[i])
