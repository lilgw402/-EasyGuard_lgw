import io

from PIL import Image
from torchvision import transforms

from easyguard import AutoModel


def img_processer():
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
    )
    trans = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return trans


vit_giant = AutoModel.from_pretrained("/mnt/bn/fashionproductxl/weights/eva_vit_giant")

# load image
black_image = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x03\x00\x00\x00\x03\x08\x02\x00\x00\x00"
    b'\xd9J"\xe8\x00\x00\x00\x12IDAT\x08\x1dcd\x80\x01F\x06\x18`d\x80\x01\x00\x00Z\x00'
    b"\x04we\x03N\x00\x00\x00\x00IEND\xaeB`\x82"
)
img = Image.open(io.BytesIO(black_image)).convert("RGB")

#
procosser = img_processer()
black_img = procosser(img)
black_input = black_img.reshape(1, 3, 224, 224)
print(f"black_input: {black_input}")

# or random_input
# rand_input = torch.randn(1, 3, 224, 224)

out = vit_giant(black_input)
print(out)
