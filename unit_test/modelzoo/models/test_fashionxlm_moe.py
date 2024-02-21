import unittest

# test module

TEST_FLAGS = ["all"]


class TestFashionXLM(unittest.TestCase):
    @unittest.skipUnless("all" in TEST_FLAGS or "fashionxlm_moe" in TEST_FLAGS, "just do it")
    def test_fashionxlm_moe(self):
        from easyguard import AutoModel, AutoTokenizer

        text = "good good study, day day up!"
        archive = "fashionxlm-moe-base"
        tokenizer = AutoTokenizer.from_pretrained(archive)
        inputs = tokenizer(text, return_tensors="pt", max_length=84)
        model = AutoModel.from_pretrained(archive, model_cls="sequence_model")
        ouputs = model(**inputs, language=["GB"])
        print(ouputs)


if __name__ == "__main__":
    unittest.main()
