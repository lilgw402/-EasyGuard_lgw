import base64
import io
import json
import os
import random
import re

import emoji
import numpy as np
import torch
import torchvision.transforms as transforms
from cruise.data_module import CruiseDataModule
from PIL import Image, ImageFile
from transformers import AutoTokenizer

from .dist_dataset import DistLineReadingDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
ImageFile.LOAD_TRUNCATED_IMAGES = True

_whitespace_re = re.compile(r"\s+")


def text_preprocess(text):
    if text is None:
        return text

    try:
        text = text.replace("\n", " ").replace("\t", " ").replace('"', " ").replace("\\", " ").strip()
        text = re.sub(r"\<.*?\>", " ", text)
        text = emoji.replace_emoji(text, replace=" ")
        text = re.sub(r"(.)\1{5,}", r"\1", text)
        text = re.sub(_whitespace_re, " ", text)
        return text
    except Exception as e:
        print(f"error occurred during cleaning: {e}")
        return text


def text_concat(title, desc=None):
    title = text_preprocess(title)
    desc = text_preprocess(desc)
    if desc is None:
        return title

    if desc.startswith(title):
        return desc

    text = title + ". " + desc

    return text


class TorchvisionLabelDataset(DistLineReadingDataset):
    def __init__(self, config, data_path, rank=0, world_size=1, shuffle=True, repeat=False, is_training=False):
        super().__init__(data_path, rank, world_size, shuffle, repeat)

        self.config = config
        self.world_size = world_size
        self.is_training = is_training
        self.text_len = config.text_len
        self.frame_len = config.frame_len
        self.head_num = config.head_num

        self.allmap = np.load("./examples/fashionbert_xl/fashionbert_xl_ptm/allmap.npy", allow_pickle=True).item()
        self.l1l2map = np.load("./examples/fashionbert_xl/fashionbert_xl_ptm/l1l2map.npy", allow_pickle=True).item()
        # self.pipe = Pipeline.from_option(f'file:./examples/fashionbert_xl/m_albert_h512a8l12')
        # self.pad_id = 0
        # self.mask_id = 280000
        self.tokenizer = AutoTokenizer.from_pretrained("/opt/tiger/xlm-roberta-base-torch")
        self.pad_id = 1
        self.preprocess = get_transform(mode="train" if is_training else "val")

        # with hopen('./examples/fashionproduct_xl/black_image.jpeg', 'rb') as f:
        #     # hdfs://harunava/home/byte_magellan_va/user/xuqi/black_image.jpeg
        #     self.black_frame = self.preprocess(self._load_image(f.read()))

        black_image = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x03\x00\x00\x00\x03\x08\x02\x00\x00\x00"
            b'\xd9J"\xe8\x00\x00\x00\x12IDAT\x08\x1dcd\x80\x01F\x06\x18`d\x80\x01\x00\x00Z\x00'
            b"\x04we\x03N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        self.black_frame = self.preprocess(self._load_image(black_image))
        # black_frame = cv2.imread('./examples/fashionproduct_xl/black_image.jpeg')
        # self.black_frame = self.cv2transform(black_frame, return_tensor=True)

        self.country2idx = {"GB": 0, "TH": 1, "ID": 2, "VN": 3, "MY": 4, "SG": 0, "PH": 0}

    def __len__(self):
        if self.is_training:
            return self.config.train_size // self.world_size
        else:
            return self.config.val_size // self.world_size

    def __iter__(self):
        for example in self.generate():
            try:
                data_item = json.loads(example)

                # label
                cid = data_item["leaf_cid"]
                # cname = self.allmap['cid2name'][cid]
                label = self.allmap["cid2label"][str(cid)]
                # if 'multi_label' in data_item:
                #     mlabelnames = data_item['multi_label'].split(';')
                #     mcids = [self.allmap['name2cid'][n.strip()] for n in mlabelnames]
                #     # mlabels = [self.gec[cid]['label'] for cid in mcids]
                #     mlabels = [self.allmap['cid2label'][cid] for cid in mcids]
                # else:
                #     mlabels = [label]
                # label_idx = data_item['label']
                # label = int(label_idx)

                # 图像
                frames = []
                if "img_base64" in data_item:
                    images = [
                        data_item["img_base64"],
                    ]
                elif "image_b64" in data_item:
                    images = [
                        data_item["image_b64"],
                    ]
                elif "images_b64" in data_item:
                    images = data_item["images_b64"]
                else:
                    print(f"cannot find image key: {data_item.keys()}")
                    continue
                # images = data_item['images_b64']
                for img_b64 in images:
                    try:
                        image_tensor = self.image_preprocess(img_b64)
                        frames.append(image_tensor)
                        if len(frames) >= self.frame_len:
                            break
                    except Exception:
                        continue

                if len(frames) < 1:
                    continue

                # 文本
                if "text" in data_item:
                    title = data_item["text"]  # kg
                    desc = None
                elif "desc" in data_item:  # goldrush and tts
                    title = data_item["title"]
                    desc = data_item["desc"]
                else:  # tmall
                    title = random.choice(json.loads(data_item["title"]))
                    desc = None
                text = text_concat(title, desc)

                country = data_item.get("country", "GB")
                country_idx = self.country2idx[country]

                # token_ids = self.pipe.preprocess([text])[0]
                # token_ids = token_ids.asnumpy()
                # token_ids = torch.from_numpy(token_ids)

                token = self.tokenizer(
                    text,
                    max_length=self.text_len,
                    padding="max_length",
                    truncation=True,
                    return_token_type_ids=True,
                    return_tensors="pt",
                )

                token_ids, token_mask, token_type = (
                    token["input_ids"],
                    token["attention_mask"],
                    token["token_type_ids"],
                )

                input_dict = {
                    "frames": frames[: self.frame_len],  # 只取了前K帧
                    "input_ids": token_ids,
                    "input_mask": token_mask,
                    "input_seg": token_type,
                    "label": label,
                    "multi_label": [label],
                    "country_idx": country_idx,
                }

                yield input_dict

            except Exception as e:
                print(f"error in dataset: {e}")

    def collect_fn(self, data):
        frames = []
        frames_mask = []
        labels = []
        input_ids = []
        input_mask = []
        input_seg = []
        multi_labels = []
        head_mask = []

        for ib, ibatch in enumerate(data):
            labels.append(ibatch["label"])
            input_ids.append(ibatch["input_ids"])
            input_mask.append(ibatch["input_mask"])
            input_seg.append(ibatch["input_seg"])
            multi_labels.append(ibatch["multi_label"] + [-100] * (10 - len(ibatch["multi_label"])))
            head = torch.zeros(self.head_num, dtype=torch.long)
            head[ibatch["country_idx"]] = 1
            head_mask.append(head)

            img_np = ibatch["frames"]
            frames_mask_cur = []
            # 判断补帧
            if len(img_np) < self.frame_len:
                for i in range(len(img_np)):
                    frames.append(img_np[i])
                    frames_mask_cur.append(1)
                for i in range(self.frame_len - len(img_np)):
                    frames.append(self.black_frame)  # 如果这个视频没有帧，就用黑帧来替代
                    frames_mask_cur.append(0)
            else:
                for i in range(self.frame_len):
                    frames.append(img_np[i])
                    frames_mask_cur.append(1)

            frames_mask.append(frames_mask_cur)

        frames_mask = torch.tensor(frames_mask)  # [bsz, frame_num]
        frames = torch.stack(frames, dim=0)  # [bsz * frame_num, c, h, w]
        _, c, h, w = frames.shape
        bsz, frame_num = frames_mask.shape
        frames = frames.reshape(bsz, self.frame_len, c, h, w)
        labels = torch.tensor(labels)
        input_ids = torch.cat(input_ids, dim=0)
        input_mask = torch.cat(input_mask, dim=0)
        input_seg = torch.cat(input_seg, dim=0)
        input_pos = torch.tensor([list(range(input_ids.shape[-1]))]).repeat(input_ids.shape[0], 1)
        multi_labels = torch.tensor(multi_labels)
        head_mask = torch.stack(head_mask, dim=0)

        res = {
            "image": frames,
            "image_mask": frames_mask,
            "label": labels,
            "multi_labels": multi_labels,
            "input_ids": input_ids,
            "input_mask": input_mask,
            "input_seg": input_seg,
            "input_pos": input_pos,
            "head_mask": head_mask,
        }
        return res

    def image_preprocess(self, image_str):
        image = self._load_image(self.b64_decode(image_str))
        image_tensor = self.preprocess(image)
        return image_tensor

    @staticmethod
    def b64_decode(string):
        if isinstance(string, str):
            string = string.encode()
        return base64.decodebytes(string)

    @staticmethod
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


class FacDataModule(CruiseDataModule):
    def __init__(
        self,
        train_files: str = None,
        train_size: int = 1500000,
        val_files: str = None,
        val_size: int = 16000,
        train_batch_size: int = 64,
        val_batch_size: int = 32,
        num_workers: int = 8,
        text_len: int = 256,
        frame_len: int = 1,
        head_num: int = 5,
        download_files: list = [],
    ):
        super().__init__()
        self.save_hparams()

    def local_rank_zero_prepare(self) -> None:
        # download cutter resource
        if self.hparams.download_files:
            to_download = [df.split("->") for df in self.hparams.download_files]
            for src, tar in to_download:
                if not os.path.exists(tar):
                    os.makedirs(tar)
                fdname = src.split("/")[-1]
                if os.path.exists(f"{tar}/{fdname}"):
                    print(f"{tar}/{fdname} already existed, pass!")
                else:
                    print(f"downloading {src} to {tar}")
                    os.system(f"hdfs dfs -get {src} {tar}")

    def setup(self, stage) -> None:
        self.train_dataset = TorchvisionLabelDataset(
            self.hparams,
            data_path=self.hparams.train_files,
            rank=int(os.environ.get("RANK") or 0),
            world_size=int(os.environ.get("WORLD_SIZE") or 1),
            shuffle=True,
            repeat=True,
            is_training=True,
        )
        print(f"len of trainset: {len(self.train_dataset)}")

        self.val_dataset = TorchvisionLabelDataset(
            self.hparams,
            data_path=self.hparams.val_files,
            rank=int(os.environ.get("RANK") or 0),
            world_size=int(os.environ.get("WORLD_SIZE") or 1),
            # world_size=1,
            shuffle=False,
            repeat=False,
            is_training=False,
        )

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.train_dataset.collect_fn,
        )
        print(len(train_loader))
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.val_batch_size,
            num_workers=1,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.train_dataset.collect_fn,
        )
        return val_loader
