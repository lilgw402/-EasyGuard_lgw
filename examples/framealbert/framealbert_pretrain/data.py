import base64
import io
import json
import os
import random
import re
from copy import deepcopy

import emoji
import numpy as np
import torch
import torchvision.transforms as transforms
from cruise.data_module import CruiseDataModule, create_cruise_loader, customized_processor
from cruise.utilities.hdfs_io import hopen
from PIL import Image, ImageFile
from ptx.matx.pipeline import Pipeline

from .dist_dataset import DistLineReadingDataset

# import albumentations as A
# from albumentations.pytorch import ToTensorV2


ImageFile.LOAD_TRUNCATED_IMAGES = True

_whitespace_re = re.compile(r"\s+")


def rmEmoji(line):
    return emoji.replace_emoji(line, replace=" ")


def rmRepeat(line):
    return re.sub(r"(.)\1{5,}", r"\1", line)


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def text_preprocess(text):
    if text is None:
        return text

    try:
        text = text.replace("\n", " ").replace("\t", " ").replace('"', "").replace("\\", "").strip()
        text = re.sub(r"\<.*?\>", " ", text)
        text = collapse_whitespace(rmRepeat(rmEmoji(text)))
        return text
    except Exception:
        return text


class TorchvisionLabelDataset(DistLineReadingDataset):
    """
    dataset，继承的dataset是 DistLineReadingDataset，是一个逐行读hdfs数据的IterableDataset。
    """

    def __init__(
        self,
        config,
        data_path,
        rank=0,
        world_size=1,
        shuffle=True,
        repeat=False,
        is_training=False,
    ):
        super().__init__(data_path, rank, world_size, shuffle, repeat)

        self.config = config
        self.world_size = world_size
        self.is_training = is_training
        self.text_len = config.text_len
        self.frame_len = config.frame_len
        self.frame_root = config.frame_root
        self.pipe = Pipeline.from_option(f"file:./examples/framealbert/m_albert_h512a8l12")  # 128
        self.mask_id = 28000
        self.pad_id = 0
        self.preprocess = get_transform(mode="train" if is_training else "val")

        with hopen("./examples/fashionproduct_xl/black_image.jpeg", "rb") as f:
            # hdfs://harunava/home/byte_magellan_va/user/xuqi/black_image.jpeg
            self.black_frame = self.preprocess(self._load_image(f.read()))

    def __len__(self):
        # world_size = os.environ.get('WORLD_SIZE') if os.environ.get('WORLD_SIZE') is not None else 1
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
            # add sep <s>
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

                # 图像
                frames = []
                frames_raw = data_item["selected_frames"]
                num_frames = len(frames_raw)
                if num_frames <= self.frame_len:
                    select_inds = list(range(num_frames))
                else:
                    step = num_frames // self.frame_len
                    select_inds = list(range(0, num_frames, step))
                    select_inds = select_inds[: self.frame_len]

                for ind in select_inds:
                    image_tensor = self.local_image_preprocess(os.path.join(self.frame_root, frames_raw[ind]))
                    frames.append(image_tensor)
                    # try:
                    #     image_tensor = self.image_preprocess(os.path.join(self.frame_root, frame_path))
                    #     frames.append(image_tensor)
                    # except:
                    #     continue

                # 文本
                video_desp = text_preprocess(data_item["video_desp"])
                product_title = text_preprocess(data_item["product_title"])
                anchor_title = text_preprocess(data_item["anchor_title"])
                merge_ocr = text_preprocess(data_item["merge_ocr"])

                texts = [video_desp, product_title, anchor_title, merge_ocr]
                # lenlimit = (96, 48, 96, 256)
                token_ids, token_seg = self.texts2tokenid(texts, padding=True)
                token_ids = np.array([token_ids], dtype=np.int64)
                token_seg = np.array([token_seg], dtype=np.int64)

                # MLM
                text_len = (token_ids == 0).argmax(axis=1)[0]
                inputs = np.array(deepcopy(token_ids), dtype=np.int64)
                input_labels = token_ids[:, :text_len]

                special_tokens_mask = input_labels < 3  # do not mask special_tokens
                # 1 indicates mask
                mask_matrix = np.random.binomial(n=1, p=0.4, size=input_labels.shape) & (special_tokens_mask == 0)

                # 不会被Mask掉的位置label设置成-100；
                input_labels[mask_matrix == 0] = -100

                # 80%的情况下，使用Mask的token来替换；
                indices_replace = (np.random.binomial(n=1, p=0.8, size=input_labels.shape) == 1) & (mask_matrix == 1)

                inputs[:, :text_len][indices_replace] = self.mask_id

                # 10%的情况下，使用随机词替换；剩余10%保持不变；
                indices_random = (
                    (mask_matrix == 1)
                    & (~indices_replace)
                    & (np.random.binomial(n=1, p=0.5, size=input_labels.shape) == 1)
                )
                random_words = np.random.randint(
                    0,
                    self.config.vocab_size - 1,
                    input_labels.shape,
                    dtype=np.int64,
                )
                inputs[:, :text_len][indices_random] = random_words[indices_random]

                inputs = torch.from_numpy(inputs)
                input_labels = np.pad(
                    input_labels,
                    ((0, 0), (0, inputs.shape[1] - text_len)),
                    "constant",
                    constant_values=(-100),
                )

                input_dict = {
                    "frames": frames,
                    "input_labels": input_labels,
                    "input_ids": inputs,
                    "input_segment_ids": token_seg,
                }

                yield input_dict

            except Exception as e:
                print(f"encounter broken data: {e}")

    def collect_fn(self, data):
        frames = []
        frames_mask = []
        itm_labels = []
        input_ids = []
        input_labels = []
        input_segment_ids = []

        for ib, ibatch in enumerate(data):
            itm_labels.append(1)
            input_ids.append(ibatch["input_ids"])
            input_labels.append(ibatch["input_labels"])
            input_segment_ids.append(torch.from_numpy(ibatch["input_segment_ids"]))

            img_np = ibatch["frames"]
            frames_mask_cur = []
            # 判断补帧
            if len(img_np) < self.frame_len:
                # print('encouter not %s frames: %s ' % (self.frame_len, len(img_np)))
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

        # construct negative samples
        nums = len(data)
        text_len = data[0]["input_ids"].shape[1]
        for ib, ibatch in enumerate(data):
            itm_labels.append(0)
            neg_id = random.choice([i for i in range(nums) if i != ib])
            input_ids.append(data[neg_id]["input_ids"])
            fake_label = np.array([-100] * text_len, dtype=np.int64)
            input_labels.append([fake_label])
            input_segment_ids.append(torch.from_numpy(ibatch["input_segment_ids"]))

            img_np = ibatch["frames"]
            frames_mask_cur = []
            # 判断补帧
            if len(img_np) < self.frame_len:
                # print('encouter not %s frames: %s ' % (self.frame_len, len(img_np)))
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
        itm_labels = torch.tensor(itm_labels)

        input_labels = torch.tensor(input_labels)
        input_labels = torch.squeeze(input_labels)
        input_ids = torch.cat(input_ids, dim=0)
        input_mask = (input_ids != 0).int()
        input_segment_ids = torch.cat(input_segment_ids, dim=0)

        res = {
            "frames": frames,
            "frames_mask": frames_mask,
            "label": itm_labels,
            "input_labels": input_labels,
            "input_ids": input_ids,
            "input_mask": input_mask,
            "input_segment_ids": input_segment_ids,
        }
        return res

    def local_image_preprocess(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image)
        return image_tensor

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
        vocab_size: int = 280001,
        text_len: int = 128,
        frame_len: int = 1,
        frame_root: str = "",
        exp: str = "default",
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
