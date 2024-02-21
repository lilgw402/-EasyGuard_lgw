import io
import json

import numpy as np
import torch
from cruise.data_module import CruiseDataModule, create_cruise_loader, customized_processor
from PIL import Image

from easyguard import AutoProcessor

# from cruise.data_module.preprocess.decode import save_args


# @save_args
class HighQualityLiveDataDecode:
    def __init__(self, key_mapping=None):
        self.key_mapping = key_mapping

    def __call__(self, data):
        if not self.key_mapping:
            return json.loads(data.decode("utf-8"))
        else:
            data_dict = json.loads(data.decode("utf-8"))
            for key in self.key_mapping:
                data_dict[self.key_mapping[key]] = data_dict[key]
                del data_dict[key]
            return data_dict


class HighQualityLiveProcessor:
    def __init__(
        self,
        test_mode,
        vocab_file="hdfs://haruna/home/byte_search_nlp_lq/multimodal/modelhub/albert_6l_zh_mix_oldcut_20200921/archer/zh_old_cut_145607.vocab",
        ocr_max_len=128,
        asr_max_len=128,
        frame_len=8,
        text_keys=("asr_text", "ocr_text"),
        frame_key="frame_file_paths",
        label_keys=("label",),
    ):
        self.processor = AutoProcessor.from_pretrained(
            "/mnt/bn/ecom-govern-maxiangqian/dongjunwei/EasyGuard/easyguard/modelzoo/models/falbert_new/config"
        )
        self.text_keys = text_keys
        self.frame_key = frame_key
        self.label_keys = label_keys
        self.black_frame = self.processor(image=Image.new("RGB", (256, 256), (255, 255, 255)))["pixel_values"]
        self.PAD = self.processor.PAD
        self.frame_len = frame_len

    def transform(self, data_dict: dict):
        # parse frame input
        frame_paths = data_dict[self.frame_key]

        try:
            item_id = frame_paths[0].split("/")[-2]
        except:
            item_id = None
        results = self.processor(
            text={
                "text_ocr": data_dict["ocr_text"],
                "text_asr": data_dict["asr_text"],
            },
            image=frame_paths,
        )

        ret = {
            "token_ids": results["token_ids"],
            "frames": results["pixel_values"] if "pixel_values" in results else [],
            "item_id": item_id,
        }
        # parse label
        labels = {}
        for lk in self.label_keys:
            labels[lk] = data_dict[lk]
        ret.update(labels)
        return ret

    def batch_transform(self, data):
        labels = []
        input_ids = []
        input_mask = []
        input_segment_ids = []
        frames = []
        frames_mask = []
        item_id = []

        max_len = max([len(b["token_ids"]) for b in data])
        for ib, ibatch in enumerate(data):
            labels.append(ibatch["label"])
            item_id.append(ibatch["item_id"])

            input_ids.append(ibatch["token_ids"][:max_len] + [self.PAD] * (max_len - len(ibatch["token_ids"])))
            input_mask.append([1] * len(ibatch["token_ids"][:max_len]) + [0] * (max_len - len(ibatch["token_ids"])))
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

        batch = {
            "frames": torch.stack(frames, dim=0),
            "frames_mask": torch.tensor(frames_mask),
            "input_ids": torch.tensor(input_ids),
            "input_mask": torch.tensor(input_mask),
            "input_segment_ids": torch.tensor(input_segment_ids),
            "label": torch.tensor(labels, dtype=torch.long),
            "item_id": item_id,
        }

        return batch


class HighQualityLiveDataModule(CruiseDataModule):
    def __init__(
        self,
        train_files: str = None,
        val_files: str = None,
        train_batch_size: int = 64,
        val_batch_size: int = 32,
        num_workers: int = 24,
        ocr_max_len: int = 256,
        asr_max_len: int = 256,
        frame_len: int = 8,
    ):
        super().__init__()
        self.save_hparams()

    def local_rank_zero_prepare(self) -> None:
        pass

    def setup(self, stage) -> None:
        self.train_files = self.hparams.train_files
        self.val_files = self.hparams.val_files

    def train_dataloader(self):
        return create_cruise_loader(
            data_sources=self.train_files,
            data_types="kv",
            batch_sizes=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            num_readers=4,
            shuffle=True,
            # predefined_steps=2000,
            processors=HighQualityLiveProcessor(
                False,
                ocr_max_len=self.hparams.ocr_max_len,
                asr_max_len=self.hparams.asr_max_len,
                frame_len=self.hparams.frame_len,
            ),
            decode_fn_list=[HighQualityLiveDataDecode()],
        )

    def val_dataloader(self):
        return create_cruise_loader(
            data_sources=self.val_files,
            data_types="kv",
            batch_sizes=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers // 2,
            num_readers=4,
            shuffle=False,
            # predefined_steps=2000,
            processors=HighQualityLiveProcessor(
                True,
                ocr_max_len=self.hparams.ocr_max_len,
                asr_max_len=self.hparams.asr_max_len,
                frame_len=self.hparams.frame_len,
            ),
            decode_fn_list=[HighQualityLiveDataDecode()],
        )

    def predict_dataloader(self, data_source=None, batch_size=32, num_workers=8):
        return create_cruise_loader(
            data_sources=data_source,
            data_types="kv",
            batch_sizes=batch_size,
            num_workers=num_workers,
            num_readers=4,
            shuffle=False,
            processors=HighQualityLiveProcessor(
                True,
                ocr_max_len=self.hparams.ocr_max_len,
                asr_max_len=self.hparams.asr_max_len,
                frame_len=self.hparams.frame_len,
            ),
            decode_fn_list=[HighQualityLiveDataDecode()],
            drop_last=False,
        )
