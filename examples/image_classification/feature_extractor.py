# --------------------------------------------------------
# AutoModel Infer
# Swin-Transformer
# Copyright (c) 2023 EasyGuard
# Written by yangmin.priv
# --------------------------------------------------------

import torch
from PIL import Image

from easyguard import AutoImageProcessor, AutoModel

model = AutoModel.from_pretrained("fashion-swin-base-224-fashionvtp")
print(model)
model.eval()

dummy_input = torch.ones(1, 3, 224, 224)
dummy_output = model(dummy_input)
print(dummy_output.size())

# infer image
image = Image.open("examples/image_classification/ptms.png").convert("RGB")
image_processor = AutoImageProcessor.from_pretrained("fashion-swin-base-224-fashionvtp")

# transform_funcs = transforms.Compose([transforms.Resize(256),
#                                       transforms.CenterCrop(224),
#                                       transforms.ToTensor(),
#                                       transforms.Normalize(
#                                           mean=[0.485, 0.456, 0.406],
#                                           std=[0.229, 0.224, 0.225])
#                                      ])
# input = transform_funcs(image)
# input_tensor = input.unsqueeze(0)
input_tensor = image_processor(image).unsqueeze(0)
output = model(input_tensor)
print(output.size())
