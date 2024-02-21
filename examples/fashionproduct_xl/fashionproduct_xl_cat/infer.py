# -*- coding: utf-8 -*-
import base64
import io

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# import cv2
import json

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFile

# from transformers import AutoTokenizer
from ptx.matx.pipeline import Pipeline
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

from cruise import CruiseCLI, CruiseTrainer
from cruise.utilities.hdfs_io import hopen

from examples.fashionproduct_xl.fashionproduct_xl_cat.data import FacDataModule, text_concat
from examples.fashionproduct_xl.fashionproduct_xl_cat.model import FrameAlbertClassify

max_len = 128
gec = np.load(
    "./examples/fashionproduct_xl/fashionproduct_xl_cat/GEC_cat.npy",
    allow_pickle=True,
).item()
pipe = Pipeline.from_option(f"file:./examples/fashionproduct_xl/m_albert_h512a8l12")
# tokenizer = AutoTokenizer.from_pretrained('./examples/fashionproduct_xl/xlm-roberta-base-torch')
country2idx = {
    "GB": 0,
    "TH": 1,
    "ID": 2,
    "VN": 3,
    "MY": 4,
}
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
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif mode == "val":
        com_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        raise ValueError("mode [%s] is not in [train, val]" % mode)
    return com_transforms


preprocess = get_transform(mode="val")


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


def process(data_item: dict):
    if "text" in data_item:
        title = data_item["text"]
        desc = None
        country = data_item["country"]
        country_idx = country2idx[country]
    else:
        title = data_item["title"]
        desc = data_item["desc"]
        country = data_item["country"]
        country_idx = country2idx[country]

    text = text_concat(title, desc)

    token_ids = pipe.preprocess([text])[0]
    token_ids = token_ids.asnumpy()
    token_ids = torch.from_numpy(token_ids)

    # tokens = tokenizer([text], padding='max_length', max_length=max_len, truncation=True, return_tensors='pt')
    # token_ids = tokens['input_ids']
    # token_mask = tokens['attention_mask']

    token_mask = token_ids.clone()
    token_mask[token_ids != 0] = 1

    input_segment_ids = torch.zeros_like(token_ids, dtype=torch.long)

    head_mask = torch.zeros([1, 5], dtype=torch.long)
    head_mask[0, country_idx] = 1

    frames = []
    if "image" in data_item:
        try:
            image_tensor = image_preprocess(data_item["image"])
            # image_array = cv2transform(load_image(data_item['image']))
            # image_tensor = torch.tensor(image_array)
            # image_tensor = image_tensor.half()
            frames.append(image_tensor)
        except:
            print(f"load image base64 failed -- {data_item.get('pid', 'no pid')}")
            return None

    frames = torch.stack(frames, dim=0)
    frames = frames.reshape([1, 1, 3, 224, 224])
    frames_mask = torch.tensor([[1]])

    return (
        token_ids,
        input_segment_ids,
        token_mask,
        frames,
        frames_mask,
        head_mask,
    )


if __name__ == "__main__":
    cli = CruiseCLI(
        FrameAlbertClassify,
        trainer_class=CruiseTrainer,
        datamodule_class=FacDataModule,
        trainer_defaults={},
    )
    cfg, trainer, model, datamodule = cli.parse_args()
    print(f"finished model config init!")

    # load ckpt
    model.setup("val")
    model.cuda()
    model.eval()

    # jit load infer
    # model = torch.jit.load('./traced_model/FrameAlbertClassify.ts')
    # model.cuda()

    countries = ["GB", "TH", "ID", "VN", "MY"]
    allres = dict()
    for country in countries:
        file = f"hdfs://harunava/home/byte_magellan_va/user/wangxian/datasets/TTS_KG_TEST/test_jsonl_high_risk_1013_country/{country}_high_risk_1013.jsonl"
        # file = f'hdfs://harunava/home/byte_magellan_va/user/wangxian/datasets/TTS_KG_TEST/test_jsonl_1013_country/{country}_1013.test.jsonl'
        with hopen(file, "r") as f:
            lines = f.readlines()

        num_all = 0
        num_top1 = 0
        num_top5 = 0
        for line in tqdm(lines):
            sample = json.loads(line)
            data = process(sample)
            if data is None:
                continue
            (
                input_ids,
                input_segment_ids,
                input_mask,
                frames,
                frames_mask,
                head_mask,
            ) = data

            res = model.forward_step(
                input_ids=input_ids.cuda(),
                input_segment_ids=input_segment_ids.cuda(),
                input_mask=input_mask.cuda(),
                frames=frames.cuda(),
                frames_mask=frames_mask.cuda(),
                head_mask=head_mask.cuda(),
            )
            logits = res["logits"]

            # jit load infer
            # res = model(input_ids.cuda(),
            #             input_segment_ids.cuda(),
            #             input_mask.cuda(),
            #             frames.cuda().half(),
            #             frames_mask.cuda(),
            #             head_mask.cuda(), )
            # logits = res

            prob, pred = logits.topk(5, 1, True, True)
            labels = [int(p) for p in pred[0]]

            num_all += 1
            if gec[sample["leaf_cid"]]["label"] == labels[0]:
                num_top1 += 1
                num_top5 += 1
            elif gec[sample["leaf_cid"]]["label"] in labels:
                num_top5 += 1
            else:
                # print(gec[sample['leaf_cid']]['label'], labels)
                pass
            if num_all % 2000 == 0:
                print(f"{country} top1 acc is {num_top1 / num_all}, with samples: {num_all}")
        print(f"{country} top1 acc is {num_top1 / num_all}, top5 acc is {num_top5 / num_all}")
        allres[country] = {
            "top1": num_top1 / num_all,
            "top5": num_top5 / num_all,
        }

    for k, v in allres.items():
        print(k, v)

# python3 examples/fashionproduct_xl/fashionproduct_xl_cat/infer.py --config examples/fashionproduct_xl/fashionproduct_xl_cat/default_config.yaml
