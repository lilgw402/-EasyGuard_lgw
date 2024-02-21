import unittest

# test module

TEST_FLAGS = ["all"]


class TestFashionXLM(unittest.TestCase):
    @unittest.skipUnless("all" in TEST_FLAGS or "fashionxlm" in TEST_FLAGS, "just do it")
    def test_fashionxlm(self):
        from easyguard import AutoModel, AutoTokenizer

        text = "good good study, day day up!"

        archive = "fashionxlm-base"
        tokenizer = AutoTokenizer.from_pretrained(archive)
        inputs = tokenizer(text, return_tensors="pt", max_length=84)
        model = AutoModel.from_pretrained(archive)
        ouputs = model(**inputs)
        print(ouputs)


if __name__ == "__main__":
    unittest.main()
