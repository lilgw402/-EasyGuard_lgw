import json

import torch
import torchvision.transforms as transforms
from cruise.data_module import CruiseDataModule, create_cruise_loader
from cruise.data_module.preprocess.decode import save_args
from PIL import Image

from easyguard.appzoo.multimodal_modeling.utils import BertTokenizer


@save_args
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


# @customized_processor()
class TextProcessor:
    def __init__(
        self,
        vocab_file="zh_old_cut_145607.vocab",
        do_lower_case=True,
        tokenize_emoji=False,
        greedy_sharp=False,
        max_len={"text_ocr": 256, "text_asr": 256},
        text_types=["text_ocr", "text_asr"],
    ):
        self.tokenizer = BertTokenizer(vocab_file, do_lower_case, tokenize_emoji, greedy_sharp)
        self.max_len = {
            "text_ocr": max_len["text_ocr"],
            "text_asr": max_len["text_asr"],
        }
        self.CLS = self.tokenizer.vocab["[CLS]"]
        self.PAD = self.tokenizer.vocab["[PAD]"]
        self.SEP = self.tokenizer.vocab["[SEP]"]
        self.MASK = self.tokenizer.vocab["[MASK]"]
        self.text_types = text_types

    def __call__(self, texts):
        tokens = ["[CLS]"]
        for text_type in self.text_types:
            text = texts[text_type]
            tokens += self.tokenizer.tokenize(text)[: self.max_len[text_type] - 2] + ["[SEP]"]
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return token_ids


# @customized_processor()
class VideoFrameProcess:
    def __init__(self, test_mode=False, frame_len=8):
        self.transform = self.get_transform(test_mode)
        self.frame_len = frame_len
        self.test_mode = test_mode

    def get_transform(self, test_mode: bool = False):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if not test_mode:
            com_transforms = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            com_transforms = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        return com_transforms

    def __call__(self, image_paths):
        frames = []
        frames_raw = image_paths[: self.frame_len]
        for frame_path in frames_raw:
            try:
                image = Image.open(frame_path).convert("RGB")
            except:  # noqa: E722
                image = Image.new("RGB", (256, 256), (255, 255, 255))
            image = self.transform(image)
            frames.append(image)
        return frames


# @customized_processor(verbose=True)
class HighQualityLiveProcessor:
    def __init__(
        self,
        test_mode,
        vocab_file="hdfs://haruna/home/byte_search_nlp_lq/multimodal/modelhub/albert_6l_zh_mix_oldcut_20200921/archer/zh_old_cut_145607.vocab",  # noqa: E501
        ocr_max_len=128,
        asr_max_len=128,
        frame_len=8,
        text_keys=("asr_text", "ocr_text"),
        frame_key="frame_file_paths",
        label_keys=("label",),
    ):
        self.text_process = TextProcessor(
            vocab_file=vocab_file,
            max_len={"text_ocr": ocr_max_len, "text_asr": asr_max_len},
        )
        self.frame_process = VideoFrameProcess(test_mode, frame_len=frame_len)
        self.text_keys = text_keys
        self.frame_key = frame_key
        self.label_keys = label_keys
        self.black_frame = self.frame_process.transform(Image.new("RGB", (256, 256), (255, 255, 255)))
        self.PAD = self.text_process.PAD
        self.frame_len = frame_len

    def transform(self, data_dict: dict):
        # parse text input
        token_ids = self.text_process(
            {
                "text_ocr": data_dict["ocr_text"],
                "text_asr": data_dict["asr_text"],
            }
        )
        # parse frame input
        frame_paths = data_dict[self.frame_key]
        try:
            item_id = frame_paths[0].split("/")[-2]
        except:  # noqa: E722
            item_id = None
        frames = self.frame_process(frame_paths)
        # print('data shape: token_ids({})'.format(len(token_ids)))
        # print('data shape: token_ids({}), frames({})'.format(len(token_ids), frames.shape))
        ret = {"token_ids": token_ids, "frames": frames, "item_id": item_id}
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
