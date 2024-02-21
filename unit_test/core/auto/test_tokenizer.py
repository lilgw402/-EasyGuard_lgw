import unittest

# test module

TEST_FLAGS = ["all"]


class TestProcessingAuto(unittest.TestCase):
    @unittest.skipUnless("all" in TEST_FLAGS or "tokenizer" in TEST_FLAGS, "just do it")
    def test_tokenizer(self):
        # from transformers import AutoProcessor, CLIPModel, CLIPProcessor
        from easyguard import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("fashion-deberta-asr", if_cache=True)
        text = "[CLS]一二三[SEP]"
        token_list = tokenizer.tokenize(text)
        print(token_list)


if __name__ == "__main__":
    unittest.main()
