# -*- coding: utf-8 -*-
import base64
import io
import json
import logging
import os
import traceback

import torch
import torchvision.transforms as transforms
from cruise.data_module import CruiseDataModule

# from cruise.data_module import customized_processor
from cruise.utilities.hdfs_io import hopen
from PIL import Image

from easyguard.appzoo.multimodal_modeling.utils import BertTokenizer

from .dist_dataset import DistLineReadingDataset

log = logging.getLogger(__name__)


class ByteDriveDataModule(CruiseDataModule):
    def __init__(
        self,
        train_path: str = "/mnt/bn/ecology-multimodal-benchmark/hq_live_data/meta/train",
        val_path: str = "/mnt/bn/ecology-multimodal-benchmark/hq_live_data/meta/val",
        vocab_file: str = "hdfs://haruna/home/byte_search_nlp_lq/multimodal/modelhub/albert_6l_zh_mix_oldcut_20200921/archer/zh_old_cut_145607.vocab",  # noqa: E501
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        num_workers: int = 4,
        val_step: int = 1234,
        data_size: int = 158028,
        train_data_size: int = 158028,
        val_data_size: int = 19995,
        ocr_max_len: int = 40,
        asr_max_len: int = 360,
        frame_len: int = 8,
    ):
        super().__init__()
        self.save_hparams()

    def local_rank_zero_prepare(self) -> None:
        pass

    def setup(self):
        # every process will run this after prepare is done
        self.train_path = self.hparams.train_path
        self.val_path = self.hparams.val_path

        self.params = {
            "ocr_max_len": self.hparams.ocr_max_len,
            "asr_max_len": self.hparams.asr_max_len,
            "vocab_file": self.hparams.vocab_file,
            "frame_len": self.hparams.frame_len,
            "train_data_size": self.hparams.train_data_size,
            "val_data_size": self.hparams.val_data_size,
        }

        self.train_dataset = TorchvisionDataset(
            self.params,
            self.train_path,
            rank=int(os.environ.get("RANK") or 0),
            world_size=int(os.environ.get("WORLD_SIZE") or 1),
            shuffle=True,
            repeat=True,
            is_training=True,
        )

        self.val_dataset = TorchvisionDataset(
            self.params,
            self.val_path,
            rank=int(os.environ.get("RANK") or 0),
            world_size=int(os.environ.get("WORLD_SIZE") or 1),
            shuffle=False,
            repeat=True,
            is_training=False,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.train_dataset.collect_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.val_dataset.collect_fn,
        )


class TorchvisionDataset(DistLineReadingDataset):
    """
    dataset，继承的dataset是 DistLineReadingDataset。
    """

    def __init__(
        self,
        params,
        data_path,
        rank=0,
        world_size=1,
        shuffle=True,
        repeat=False,
        is_training=False,
    ):
        super().__init__(data_path, rank, world_size, shuffle, repeat, verbose=False)
        self.preprocess = get_transform(mode="train" if is_training else "val")

        self.max_len = {
            "text_ocr": params["ocr_max_len"],
            "text_asr": params["asr_max_len"],
        }
        self.frame_len = params["frame_len"]
        self.tokenizer = BertTokenizer(
            params["vocab_file"],
            do_lower_case=True,
            tokenize_emoji=False,
            greedy_sharp=False,
        )

        self.PAD = self.tokenizer.vocab["[PAD]"]
        with hopen(
            "hdfs://haruna/home/byte_search_nlp_lq/multimodal/black_frame.jpg",
            "rb",
        ) as f:
            self.black_frame = self.preprocess(self._load_image(f.read()))

        self.training = is_training
        self.text_types = ["text_asr", "text_ocr"]

        self.params = params

    def __iter__(self):
        for example in self.generate():
            try:
                data_item = json.loads(example)
                # 标签
                label = data_item["label"]

                # 文本
                texts = {
                    "text_ocr": data_item["ocr_text"],
                    "text_asr": data_item["asr_text"],
                }
                token_ids = self.text_preprocess(texts)

                # 多帧
                frames = []
                frames_raw = data_item["frame_file_paths"][: self.frame_len]
                for frame_path in frames_raw:
                    image_tensor = self.image_preprocess(frame_path)
                    frames.append(image_tensor)
                input_dict = {
                    "frames": frames,
                    "input_ids": token_ids,
                    "label": label,
                }
                yield input_dict

            except Exception as e:
                if self.training:
                    log.error(traceback.format_exc())
                    log.error("encounter broken training data: %s" % e)
                elif "Expecting value:" not in str(e):
                    log.error(traceback.format_exc())
                    log.error("encounter broken data: %s" % e)

    def __len__(self):
        if self.training:
            return self.params["train_data_size"] // int(os.environ.get("WORLD_SIZE"))
        else:
            return self.params["val_data_size"] // int(os.environ.get("WORLD_SIZE"))

    def collect_fn(self, data):
        labels = []
        input_ids = []
        input_mask = []
        input_segment_ids = []
        frames = []
        frames_mask = []

        max_len = max([len(b["input_ids"]) for b in data])

        for ib, ibatch in enumerate(data):
            labels.append(ibatch["label"])

            input_ids.append(ibatch["input_ids"][:max_len] + [self.PAD] * (max_len - len(ibatch["input_ids"])))
            input_mask.append([1] * len(ibatch["input_ids"][:max_len]) + [0] * (max_len - len(ibatch["input_ids"])))
            input_segment_ids.append([0] * max_len)

            frames_cur = []
            frames_mask_cur = []
            for img in ibatch["frames"]:
                frames_cur.append(img)
                frames_mask_cur.append(1)
            while len(frames_cur) < self.frame_len:
                frames_cur.append(self.black_frame)
                frames_mask_cur.append(0)
            frames.append(torch.stack(frames_cur, dim=0))
            frames_mask.append(frames_mask_cur)

        res = {
            "frames": torch.stack(frames, dim=0),
            "frames_mask": torch.tensor(frames_mask),
            "input_ids": torch.tensor(input_ids),
            "input_mask": torch.tensor(input_mask),
            "input_segment_ids": torch.tensor(input_segment_ids),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        return res

    def text_preprocess(self, texts):
        tokens = ["[CLS]"]
        for text_type in self.text_types:
            text = texts[text_type][: self.max_len[text_type] - 2]
            tokens += self.tokenizer.tokenize(text) + ["[SEP]"]
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return token_ids

    def image_preprocess(self, image_path):
        image = Image.open(image_path)
        image_tensor = self.preprocess(image)
        return image_tensor

    @staticmethod
    def b64_decode(string):
        if isinstance(string, str):
            string = string.encode()
        return base64.decodebytes(string)

    @staticmethod
    def _load_image(buffer):
        img = Image.open(io.BytesIO(buffer)).convert("RGB")
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
