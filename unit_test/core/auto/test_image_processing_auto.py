import unittest

# test module

TEST_FLAGS = ["all"]


class TestImageProcessor(unittest.TestCase):
    @unittest.skipUnless("all" in TEST_FLAGS or "hf_model" in TEST_FLAGS, "just do it")
    def test_hf_model(self):
        from easyguard import AutoImageProcessor, AutoModel

        extractor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

        model = AutoModel.from_pretrained("google/vit-base-patch16-224")
        print(extractor)
        print(model)


if __name__ == "__main__":
    unittest.main()
