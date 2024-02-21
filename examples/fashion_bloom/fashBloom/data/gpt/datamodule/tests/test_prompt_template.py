import pandas as pd
from promptsource.templates import DatasetTemplates


def test_load_docs(input_file="./ARC-Easy-Train.parquet"):
    docs = pd.read_parquet(input_file, "pyarrow")
    docs = pd.read_table(input_file)

    return docs.question[0], docs.choices[0], docs.answerKey[0]


def load_docs(input_file="./ARC-Easy-Train.parquet"):
    doc = {
        "question": "Which factor will most likely cause a person to develop a fever?",
        "choices": {
            "label": ["A", "B", "C", "D"],
            "text": [
                "a leg muscle relaxing after exercise",
                "a bacterial population in the bloodstream",
                "several viral particles on the skin",
                "carbohydrates being digested in the stomach",
            ],
        },
        "answerKey": "B",
    }

    return doc


def test_prompt(
    doc,
    dataset_name="ai2_arc",
    subset_name="ARC-Easy",
    template_name="pick_the_most_correct_option",
):
    templates = DatasetTemplates(dataset_name, subset_name)

    # print(templates)
    prompt_template = templates[template_name]
    # print(prompt)

    outputs = prompt_template.apply(doc)

    answer_choices_list = prompt_template.get_answer_choices_list(doc)

    metrics = prompt_template.metadata.metrics

    prompt, target = None, None
    if len(outputs) >= 2:
        prompt = outputs[0]
        target = outputs[1]

    print(prompt)
    print(target)
    print(answer_choices_list)
    print(metrics)


if __name__ == "__main__":
    doc = load_docs()
    test_prompt(doc)
    # print(docs)
