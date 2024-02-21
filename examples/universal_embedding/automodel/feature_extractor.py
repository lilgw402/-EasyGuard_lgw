# --------------------------------------------------------
# AutoModel Infer
# Fashion Universal
# Copyright (c) 2023 EasyGuard
# Written by yangmin.priv
# --------------------------------------------------------


import torch
from PIL import Image

from easyguard import AutoImageProcessor, AutoModel

model = AutoModel.from_pretrained("fashion-universal-vit-base-224")
# model = AutoModel.from_pretrained("fashion-universal-product-vit-base-224")
print(model)
model.eval()

dummy_input = torch.ones(1, 3, 224, 224)
dummy_output = model(dummy_input)
print(dummy_output.size())

# infer image
image = Image.open("0.jpg").convert("RGB")
image_processor = AutoImageProcessor.from_pretrained("fashion-universal-vit-base-224")

input_tensor = image_processor(image).unsqueeze(0)
output = model(input_tensor)
print(output.size())
