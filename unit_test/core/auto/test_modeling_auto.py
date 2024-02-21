import unittest

# test module

TEST_FLAGS = ["all"]


class TestModelAuto(unittest.TestCase):
    @unittest.skipUnless("all" in TEST_FLAGS or "hf_model" in TEST_FLAGS, "just do it")
    def test_hf_model(self):
        from easyguard import AutoModel, AutoTokenizer

        # archive = "bert-base-uncased"
        archive = "hfl/chinese-roberta-wwm-ext"
        tokenizer = AutoTokenizer.from_pretrained(archive)
        model = AutoModel.from_pretrained(archive)
        inputs = tokenizer("Hello world!", return_tensors="pt")
        print(inputs)
        ouputs = model(**inputs)
        print(ouputs)

    @unittest.skipUnless("all" in TEST_FLAGS or "default_model" in TEST_FLAGS, "just do it")
    def test_default_model(self):
        import torch

        from easyguard.core import AutoModel, AutoTokenizer

        archive = "fashion-deberta-ccr-order"
        my_model = AutoModel.from_pretrained(archive, dim_shrink=128)
        my_tokenizer = AutoTokenizer.from_pretrained(archive)
        my_model.eval()
        max_length = 512
        text = "我的手机"
        text_token = my_tokenizer.tokenize(text)
        text_token = ["[CLS]"] + text_token + ["[SEP]"]
        token_ids = my_tokenizer.convert_tokens_to_ids(text_token)
        input_ids = token_ids[:max_length] + my_tokenizer.convert_tokens_to_ids(["[PAD]"]) * (
            max_length - len(token_ids)
        )
        input_mask = [1] * len(token_ids[:max_length]) + [0] * (max_length - len(token_ids))
        input_segment_ids = [0] * max_length
        print(f"input_ids: {input_ids}")
        print(f"input_mask: {input_mask}")
        print(f"input_segment_ids: {input_segment_ids}")
        input_ids = torch.tensor([input_ids])
        input_mask = torch.tensor([input_mask])
        input_segment_ids = torch.tensor([input_segment_ids])
        input_ids, input_mask, input_segment_ids = my_tokenizer(text).values()
        with torch.no_grad():
            result = my_model(
                input_ids=input_ids,
                attention_mask=input_mask,
                segment_ids=input_segment_ids,
                output_pooled=True,
            )
        print(result)

    @unittest.skipUnless("all" in TEST_FLAGS or "hf_modified_model" in TEST_FLAGS, "just do it")
    def test_hf_modified_model(self):
        from easyguard.core import AutoModel, AutoTokenizer

        text = "good good study, day day up!"

        # archive = "fashionxlm-base"
        # tokenizer = AutoTokenizer.from_pretrained(archive)
        # inputs = tokenizer(text, return_tensors="pt", max_length=84)
        # model = AutoModel.from_pretrained(archive)
        # ouputs = model(**inputs)
        # print(ouputs)

        archive = "fashionxlm-moe-base"
        tokenizer = AutoTokenizer.from_pretrained(archive)
        inputs = tokenizer(text, return_tensors="pt", max_length=84)
        model = AutoModel.from_pretrained(archive)
        ouputs = model(**inputs, language=["GB"])
        print(ouputs)

        archive = "fashionxlm-moe-base"
        tokenizer = AutoTokenizer.from_pretrained(archive)
        inputs = tokenizer(text, return_tensors="pt", max_length=84)
        model = AutoModel.from_pretrained(archive, model_cls="sequence_model")
        ouputs = model(**inputs, language=["GB"])
        print(ouputs)

        archive = "xlmr-base"
        tokenizer = AutoTokenizer.from_pretrained(archive)
        inputs = tokenizer(text, return_tensors="pt", max_length=84)
        model = AutoModel.from_pretrained(archive)

        ouputs = model(**inputs)
        print(ouputs)

    @unittest.skipUnless("all" in TEST_FLAGS or "local_defult_model" in TEST_FLAGS, "just do it")
    def test_local_defult_model(self):
        import torch

        from easyguard import AutoModel, AutoTokenizer

        archive = (
            "/root/.cache/easyguard/models/deberta/6ca54229f34a5b0936d549632d9566a9a07be8ac430528306e27566964dc6a4a"
        )
        my_model = AutoModel.from_pretrained(archive, dim_shrink=128)
        my_tokenizer = AutoTokenizer.from_pretrained(archive)
        my_model.eval()
        max_length = 512
        text = "我的手机"
        text_token = my_tokenizer.tokenize(text)
        text_token = ["[CLS]"] + text_token + ["[SEP]"]
        token_ids = my_tokenizer.convert_tokens_to_ids(text_token)
        input_ids = token_ids[:max_length] + my_tokenizer.convert_tokens_to_ids(["[PAD]"]) * (
            max_length - len(token_ids)
        )
        input_mask = [1] * len(token_ids[:max_length]) + [0] * (max_length - len(token_ids))
        input_segment_ids = [0] * max_length
        print(f"input_ids: {input_ids}")
        print(f"input_mask: {input_mask}")
        print(f"input_segment_ids: {input_segment_ids}")
        input_ids = torch.tensor([input_ids])
        input_mask = torch.tensor([input_mask])
        input_segment_ids = torch.tensor([input_segment_ids])
        input_ids, input_mask, input_segment_ids = my_tokenizer(text).values()
        with torch.no_grad():
            result = my_model(
                input_ids=input_ids,
                attention_mask=input_mask,
                segment_ids=input_segment_ids,
                output_pooled=True,
            )
        print(result)

        archive = "/root/.cache/easyguard/models/fashionxlm_moe/6c2f5988fb7ea932b4914cf0fc6c1acb2460de2ee93f2a31370fa9d45f070f37"  # noqa: E501
        tokenizer = AutoTokenizer.from_pretrained(archive)
        inputs = tokenizer(text, return_tensors="pt", max_length=84)
        model = AutoModel.from_pretrained(archive, model_cls="sequence_model")
        ouputs = model(**inputs, language=["GB"])
        print(ouputs)


if __name__ == "__main__":
    # test_hf_model()
    unittest.main()
