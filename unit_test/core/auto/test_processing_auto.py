import unittest

import requests
from PIL import Image

# test module

TEST_FLAGS = ["all"]


class TestProcessingAuto(unittest.TestCase):
    @unittest.skipUnless("all" in TEST_FLAGS or "original_clip" in TEST_FLAGS, "just do it")
    def test_original_clip(self):
        # from transformers import AutoProcessor, CLIPModel, CLIPProcessor
        from easyguard import AutoModel, AutoProcessor

        archive = "/root/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/e6a30b603a447e251fdaca1c3056b2a16cdfebeb"
        # archive = "openai/clip-vit-base-patch32"
        model = AutoModel.from_pretrained(archive)
        processor = AutoProcessor.from_pretrained(archive)

        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        inputs = processor(
            text=["a photo of a cat", "a photo of a dog", "a photo of kitty"],
            images=image,
            return_tensors="pt",
            padding=True,
        )

        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilitiesx = 1

        print(probs)

    @unittest.skipUnless("all" in TEST_FLAGS or "vit" in TEST_FLAGS, "just do it")
    def test_vit(self):
        from easyguard import AutoImageProcessor, AutoModel

        extractor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

        model = AutoModel.from_pretrained("google/vit-base-patch16-224")

    @unittest.skipUnless("all" in TEST_FLAGS or "clip" in TEST_FLAGS, "just do it")
    def test_clip(self):
        from easyguard import AutoModel, AutoProcessor

        path = "openai/clip-vit-base-patch32"
        model = AutoModel.from_pretrained(path)
        processor = AutoProcessor.from_pretrained(path)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        inputs = processor(
            text=["a photo of a cat", "a photo of a dog", "a photo of kitty"],
            images=image,
            return_tensors="pt",
            padding=True,
        )

        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilitiesx = 1
        print(probs)

    @unittest.skipUnless("all" in TEST_FLAGS or "falbert" in TEST_FLAGS, "just do it")
    def test_falbert(self):
        from easyguard import AutoModel, AutoProcessor

        archive = "falbert-no-weights"
        processor = AutoProcessor.from_pretrained(archive)


if __name__ == "__main__":
    unittest.main()
