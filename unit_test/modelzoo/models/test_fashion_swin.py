import unittest

# test module

TEST_FLAGS = ["fashion_swin"]


class TestFashionSwin(unittest.TestCase):
    @unittest.skipUnless("all" in TEST_FLAGS or "fashion_swin" in TEST_FLAGS, "just do it")
    def test_fashion_swin(self):
        import torch

        from easyguard import AutoImageProcessor, AutoModel

        dummy_input = torch.ones(1, 3, 680, 728)
        archive = "fashion-swin-base-224-fashionvtp"
        image_processor = AutoImageProcessor.from_pretrained(archive)
        inputs = image_processor(dummy_input)
        model = AutoModel.from_pretrained(archive, if_cache=True)
        model.eval()
        ouputs = model(inputs)
        print(ouputs)

    @unittest.skipUnless(
        "all" in TEST_FLAGS or "fashion_swin_local_dir" in TEST_FLAGS,
        "just do it",
    )
    def test_fashion_swin_local_dir(self):
        import torch

        from easyguard import AutoImageProcessor, AutoModel

        dummy_input = torch.ones(1, 3, 680, 728)
        archive = "/root/.cache/easyguard/models/fashion_swin/0007f434eb731c6f5a799d7b773390dd6ab319bc97e314a17cac5254365b502a"  # noqa: E501
        image_processor = AutoImageProcessor.from_pretrained(archive)
        inputs = image_processor(dummy_input)
        model = AutoModel.from_pretrained(archive)
        model.eval()
        ouputs = model(inputs)
        print(ouputs)

    @unittest.skipUnless(
        "all" in TEST_FLAGS or "fashion_swin_local_file" in TEST_FLAGS,
        "just do it",
    )
    def test_fashion_swin_local_file(self):
        import torch

        from easyguard import AutoImageProcessor, AutoModel

        dummy_input = torch.ones(1, 3, 680, 728)
        archive = "/root/.cache/easyguard/models/fashion_swin/0007f434eb731c6f5a799d7b773390dd6ab319bc97e314a17cac5254365b502a/config.yaml"  # noqa: E501
        image_processor = AutoImageProcessor.from_pretrained(
            "/root/.cache/easyguard/models/fashion_swin/0007f434eb731c6f5a799d7b773390dd6ab319bc97e314a17cac5254365b502a/"  # noqa: E501
        )
        inputs = image_processor(dummy_input)
        model = AutoModel.from_pretrained(archive)
        model.eval()
        ouputs = model(inputs)
        print(ouputs)


if __name__ == "__main__":
    unittest.main()
