# -*- coding: utf-8 -*-
import base64
import io
import json
import re

import torch
import torchvision.transforms as transforms
from cruise import CruiseCLI, CruiseTrainer
from cruise.utilities.hdfs_io import hopen
from PIL import Image, ImageFile
from ptx.matx.pipeline import Pipeline
from tqdm import tqdm

from examples.fashionproduct_xl.fashionproduct_xl_audit.data import FacDataModule
from examples.fashionproduct_xl.fashionproduct_xl_audit.model import FPXL

ImageFile.LOAD_TRUNCATED_IMAGES = True

max_len = 512
max_frame = 3
_whitespace_re = re.compile(r"\s+")
pipe = Pipeline.from_option("file:./examples/fashionproduct_xl/m_albert_h512a8l12")
pad_id = 0
country2idx = {"GB": 0, "TH": 1, "ID": 2, "VN": 3, "MY": 4, "PH": 0, "SG": 0}

# default_mean = np.array((0.485, 0.456, 0.406)).reshape(1, 1, 1, 3)
# default_std = np.array((0.229, 0.224, 0.225)).reshape(1, 1, 1, 3)


def image_preprocess(image_str):
    image = _load_image(b64_decode(image_str))
    image_tensor = preprocess(image)
    return image_tensor


def b64_decode(string):
    if isinstance(string, str):
        string = string.encode()
    return base64.decodebytes(string)


def _load_image(buffer):
    img = Image.open(io.BytesIO(buffer))
    img = img.convert("RGB")
    return img


def get_transform(mode: str = "train"):
    """
    根据不同的data，返回不同的transform
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if mode == "train":
        com_transforms = transforms.Compose(
            [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]
        )
    elif mode == "val":
        com_transforms = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize]
        )
    else:
        raise ValueError("mode [%s] is not in [train, val]" % mode)
    return com_transforms


preprocess = get_transform(mode="val")

with hopen("./examples/fashionproduct_xl/black_image.jpeg", "rb") as f:
    black_frame = preprocess(_load_image(f.read()))


# def load_image(image_str):
#     imgString = base64.b64decode(image_str)
#     image_data = np.fromstring(imgString, np.uint8)
#     image_byte = np.frombuffer(image_data, np.int8)
#     img_cv2 = cv2.imdecode(image_byte, cv2.IMREAD_COLOR)
#     # img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
#     # return img_rgb
#     return img_cv2
#
#
# def cv2transform(img_cv2):
#     img_cv2resize = cv2.resize(img_cv2, (256, 256), interpolation=cv2.INTER_AREA)
#     img_crop = img_cv2resize[16:240, 16:240]
#     img_cv22np = np.asarray(img_crop)[np.newaxis, :, :, ::-1]
#     img_cv22np = (img_cv22np / 255.0 - default_mean) / default_std
#     img_cv22np_transpose = img_cv22np.transpose(0, 3, 1, 2)
#     # img_half = img_cv22np_transpose.astype(np.float16)
#     img_array = img_cv22np_transpose.astype(np.float32)
#
#     return img_array


def text_preprocess(text):
    if text is None:
        return text

    try:
        text = text.replace("\n", " ").replace("\t", " ").replace('"', "").replace("\\", "").strip()
        text = re.sub(r"\<.*?\>", " ", text)
        # text = emoji.replace_emoji(text, replace=' ')
        text = re.sub(r"(.)\1{5,}", r"\1", text)
        text = re.sub(_whitespace_re, " ", text)
        return text
    except Exception as e:
        print(f"error in text_preprocess: {e}")
        return text


def texts2tokenid(texts, padding, lenlimit=None):
    # todo, use lenlimit
    token_ids = []
    token_seg = []
    for idx, t in enumerate(texts):
        if idx == 0:
            tokens = [i for i in pipe.preprocess([t])[0][0].to_list() if i > 1]  # pad, cls and sep not include
        else:
            tokens = [i for i in pipe.preprocess([t])[0][0].to_list() if i > 2]  # pad, cls and sep not include
        seg = [idx] * len(tokens)

        token_ids += tokens
        token_seg += seg
        token_ids += [1]
        token_seg += [idx]

    if len(token_ids) < lenlimit and padding:
        token_ids += [pad_id] * (lenlimit - len(token_ids))
        token_seg += [0] * (lenlimit - len(token_seg))

    else:
        token_ids = token_ids[:lenlimit]
        token_seg = token_seg[:lenlimit]

    token_mask = [1 if i != pad_id else 0 for i in token_ids]

    return token_ids, token_seg, token_mask


def process(data_item: dict):
    title = " ".join(data_item["product_name"])
    brands = " ".join(data_item["brands"])
    attr = " ".join(data_item["key_attribute"])
    cate = " ".join(data_item["category"])
    desc = " ".join(data_item["description_text"])
    country = data_item.get("country", 0)
    country_idx = country2idx[country]

    texts = [title, brands, attr, cate, desc]
    token_ids, token_seg, token_mask = texts2tokenid(texts, padding=True, lenlimit=max_len)
    token_ids = torch.tensor([token_ids], dtype=torch.long)
    token_seg = torch.tensor([token_seg], dtype=torch.long)
    token_mask = torch.tensor([token_mask], dtype=torch.long)

    # token_mask = token_ids.clone()
    # token_mask[token_ids != pad_id] = 1

    head_mask = torch.zeros([1, 5], dtype=torch.long)
    head_mask[0, country_idx] = 1

    frames = []
    frames_mask_cur = []
    all_images = data_item["images_b64"] + data_item["description_images_b64"]
    all_images = []
    if all_images:
        # get image by base64
        for images in all_images:
            image_tensor = image_preprocess(images)
            frames.append(image_tensor)
            frames_mask_cur.append(1)
    for _ in range(max_frame - len(frames)):
        frames.append(black_frame)  # 如果这个视频没有帧，就用黑帧来替代
        frames_mask_cur.append(0)

    frames = torch.stack(frames, dim=0)
    frames = frames.reshape([1, max_frame, 3, 224, 224])
    frames_mask = torch.tensor([frames_mask_cur])

    return token_ids, token_seg, token_mask, frames, frames_mask, head_mask


if __name__ == "__main__":
    cli = CruiseCLI(FPXL, trainer_class=CruiseTrainer, datamodule_class=FacDataModule, trainer_defaults={})
    cfg, trainer, model, datamodule = cli.parse_args()
    print("finished model config init!")

    # load ckpt
    model.setup("val")
    model.cuda()
    model.eval()

    # jit load infer
    # model = torch.jit.load('./traced_model/FrameAlbertClassify.ts')
    # model.cuda()

    with open("audit_test.jsonl", "r") as f:
        lines = f.readlines()

    for line in tqdm(lines[:1]):
        sample = json.loads(line)
        data = process(sample)
        input_ids, input_segment_ids, input_mask, frames, frames_mask, head_mask = data
        for i in data:
            print(i.shape)

        res = model.forward_step(
            input_ids=input_ids.cuda(),
            input_segment_ids=input_segment_ids.cuda(),
            input_mask=input_mask.cuda(),
            frames=frames.cuda(),
            frames_mask=frames_mask.cuda(),
            head_mask=head_mask.cuda(),
        )
        logits = res["logits"]
        print(logits)

#         # jit load infer
#         # res = model(input_ids.cuda(),
#         #             input_segment_ids.cuda(),
#         #             input_mask.cuda(),
#         #             frames.cuda().half(),
#         #             frames_mask.cuda(),
#         #             head_mask.cuda(), )
#         # logits = res

#         prob, pred = logits.topk(5, 1, True, True)
#         labels = [int(p) for p in pred[0]]

#         num_all += 1
#         if gec[sample['leaf_cid']]['label'] == labels[0]:
#             num_top1 += 1
#             num_top5 += 1
#         elif gec[sample['leaf_cid']]['label'] in labels:
#             num_top5 += 1
#         else:
#             # print(gec[sample['leaf_cid']]['label'], labels)
#             pass
#         if num_all % 2000 == 0:
#             print(f'{country} top1 acc is {num_top1 / num_all}, with samples: {num_all}')
#     print(f'{country} top1 acc is {num_top1 / num_all}, top5 acc is {num_top5 / num_all}')
#     allres[country] = {'top1': num_top1 / num_all, 'top5': num_top5 / num_all}

# for k, v in allres.items():
#     print(k, v)

# python3 examples/fashionproduct_xl/fashionproduct_xl_audit/infer.py
# --config examples/fashionproduct_xl/fashionproduct_xl_audit/default_config.yaml
