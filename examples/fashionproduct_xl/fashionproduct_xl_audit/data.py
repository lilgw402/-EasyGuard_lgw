import base64
import io
import json
import os
import random
import re

import emoji
import torch
import torchvision.transforms as transforms
from cruise.data_module import CruiseDataModule
from cruise.utilities.hdfs_io import hopen
from PIL import Image, ImageFile
from ptx.matx.pipeline import Pipeline

from .dist_dataset import DistLineReadingDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

_whitespace_re = re.compile(r"\s+")


def text_preprocess(text):
    if text is None:
        return text

    try:
        text = text.replace("\n", " ").replace("\t", " ").replace('"', "").replace("\\", "").strip()
        text = re.sub(r"\<.*?\>", " ", text)
        text = emoji.replace_emoji(text, replace=" ")
        text = re.sub(r"(.)\1{5,}", r"\1", text)
        text = re.sub(_whitespace_re, " ", text)
        return text
    except Exception as e:
        print(f"error occurred during cleaning: {e}")
        return text


class TorchvisionLabelDataset(DistLineReadingDataset):
    """
    dataset，继承的dataset是 DistLineReadingDataset，是一个逐行读hdfs数据的IterableDataset。
    """

    def __init__(self, config, data_path, rank=0, world_size=1, shuffle=True, repeat=False, is_training=False):
        super().__init__(data_path, rank, world_size, shuffle, repeat)

        self.config = config
        self.world_size = world_size
        self.is_training = is_training
        self.text_len = config.text_len
        self.frame_len = config.frame_len
        self.head_num = config.head_num

        self.pipe = Pipeline.from_option("file:./examples/fashionproduct_xl/m_albert_h512a8l12")
        self.pad_id = 0
        self.preprocess = get_transform(mode="train" if is_training else "val")

        with hopen("./examples/fashionproduct_xl/black_image.jpeg", "rb") as f:
            self.black_frame = self.preprocess(self._load_image(f.read()))

        self.country2idx = {"GB": 0, "TH": 1, "ID": 2, "VN": 3, "MY": 4, "PH": 0, "SG": 0}

    def __len__(self):
        if self.is_training:
            return self.config.train_size // self.world_size
        else:
            return self.config.val_size // self.world_size

    def texts2tokenid(self, texts, padding, lenlimit=None):
        # todo, use lenlimit
        token_ids = []
        token_seg = []
        for idx, t in enumerate(texts):
            if idx == 0:
                tokens = [
                    i for i in self.pipe.preprocess([t])[0][0].to_list() if i > 1
                ]  # pad, cls and sep not include
            else:
                tokens = [
                    i for i in self.pipe.preprocess([t])[0][0].to_list() if i > 2
                ]  # pad, cls and sep not include
            seg = [idx] * len(tokens)

            token_ids += tokens
            token_seg += seg
            token_ids += [1]
            token_seg += [idx]

        if len(token_ids) < self.text_len and padding:
            token_ids += [self.pad_id] * (self.text_len - len(token_ids))
            token_seg += [0] * (self.text_len - len(token_seg))
        else:
            token_ids = token_ids[: self.text_len]
            token_seg = token_seg[: self.text_len]

        return token_ids, token_seg

    def __iter__(self):
        for example in self.generate():
            try:
                data_item = json.loads(example)

                # label
                # label = int(data_item['audit_label_binary_label'])
                label = int(data_item["audit_label_multi_label"])

                # 图像
                frames = []
                all_images = data_item["images_b64"] + data_item["description_images_b64"]
                if all_images:
                    # get image by base64
                    for images in all_images:
                        try:
                            image_tensor = self.image_preprocess(images)
                            frames.append(image_tensor)
                        except Exception as e:
                            print(f"load image base64 failed -- {data_item.get('pid', 'None pid')} -- {e}")
                            continue
                else:
                    raise Exception("cannot find images")

                if len(frames) > self.frame_len:
                    # 抽取首图尾图以及中间随机m-2帧图片
                    inds = list(range(1, len(frames) - 1))
                    random.shuffle(inds)
                    choose_idx = [0] + sorted(inds[: self.frame_len - 2]) + [len(frames) - 1]
                    frames = [frames[i] for i in choose_idx]

                    # 抽取前m帧
                    # frames = frames[: self.frame_len]

                # 文本
                title = " ".join(data_item["product_name"])
                brands = " ".join(data_item["brands"])
                attr = " ".join(data_item["key_attribute"])
                cate = " ".join(data_item["category"])
                desc = " ".join(data_item["description_text"])

                texts = [title, brands, attr, cate, desc]
                token_ids, token_seg = self.texts2tokenid(texts, padding=True, lenlimit=self.text_len)
                token_ids = torch.tensor([token_ids], dtype=torch.long)
                token_seg = torch.tensor([token_seg], dtype=torch.long)

                country = data_item.get("country", "GB")
                country_idx = self.country2idx.get(country, 0)

                input_dict = {
                    "frames": frames,
                    "label": label,
                    "country_idx": country_idx,
                    "input_ids": token_ids,
                    "input_seg": token_seg,
                }

                yield input_dict

            except Exception as e:
                print(f"error in dataset: {e}")

    def collect_fn(self, data):
        frames = []
        frames_mask = []
        labels = []
        head_mask = []
        input_ids = []
        input_segs = []
        input_mask = []

        for ib, ibatch in enumerate(data):
            labels.append(ibatch["label"])
            # country_idx.append(ibatch["country_idx"])
            head = torch.zeros(self.head_num, dtype=torch.long)
            head[ibatch["country_idx"]] = 1
            head_mask.append(head)
            input_ids.append(ibatch["input_ids"])
            input_segs.append(ibatch["input_seg"])
            input_mask_id = ibatch["input_ids"].clone()
            input_mask_id[input_mask_id != self.pad_id] = 1
            input_mask.append(input_mask_id)

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
        frames = frames.reshape([bsz, frame_num, c, h, w])
        labels = torch.tensor(labels)
        head_mask = torch.stack(head_mask, dim=0)
        input_ids = torch.cat(input_ids, dim=0)
        input_segment_ids = torch.cat(input_segs, dim=0)
        input_mask = torch.cat(input_mask, dim=0)

        res = {
            "frames": frames,
            "frames_mask": frames_mask,
            "label": labels,
            "head_mask": head_mask,
            "input_ids": input_ids,
            "input_mask": input_mask,
            "input_segment_ids": input_segment_ids,
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
        text_len: int = 128,
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
