import io
import json

import numpy as np
import torch
import torchvision.transforms as transforms
from cruise.data_module import CruiseDataModule, create_cruise_loader
from cruise.data_module.preprocess.decode import save_args
from PIL import Image

from easyguard.appzoo.multimodal_modeling.utils import BertTokenizer


@save_args
class AuthenticDataDecode:
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
        max_len=256,
    ):
        self.tokenizer = BertTokenizer(
            vocab_file,
            do_lower_case,
            tokenize_emoji,
            greedy_sharp,
            max_len=max_len,
        )
        self.CLS = self.tokenizer.vocab["[CLS]"]
        self.PAD = self.tokenizer.vocab["[PAD]"]
        self.SEP = self.tokenizer.vocab["[SEP]"]
        self.MASK = self.tokenizer.vocab["[MASK]"]
        self.max_len = max_len

    def __call__(self, text: str):
        tokens = self.tokenizer.tokenize(text)
        tokens = ["[CLS]"] + tokens
        tokens = tokens[: self.max_len]
        masks = [1] * len(tokens)
        segment_ids = [0] * len(tokens)

        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        cur_len = len(token_ids)
        if self.max_len > cur_len:
            masks.extend([0] * (self.max_len - cur_len))
            segment_ids.extend([0] * (self.max_len - cur_len))
            token_ids.extend([self.PAD] * (self.max_len - cur_len))

        return token_ids, masks, segment_ids


# @customized_processor()
class VideoFrameProcess(object):
    def __init__(
        self,
        test_mode=False,
        clip_len=1,
        frame_interval=1,
        num_clips=4,
        keep_tail_frames=False,
        out_of_bound_opt="loop",
    ):
        self.transform = self.get_transform(test_mode)
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.keep_tail_frames = keep_tail_frames
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode

    def get_transform(self, test_mode: bool = False):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if not test_mode:
            com_transforms = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.01, 0.2)),
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
        total_frames = len(image_paths)
        if total_frames > 0:
            clip_offsets = self._sample_clips(total_frames)
            frame_inds = clip_offsets[:, None] + np.arange(self.clip_len)[None, :] * self.frame_interval
            frame_inds = np.concatenate(frame_inds)
            frame_inds = frame_inds.reshape((-1, self.clip_len))
            if self.out_of_bound_opt == "loop":
                frame_inds = np.mod(frame_inds, total_frames)
            elif self.out_of_bound_opt == "repeat_last":
                safe_inds = frame_inds < total_frames
                unsafe_inds = 1 - safe_inds
                last_ind = np.max(safe_inds * frame_inds, axis=1)
                new_inds = safe_inds * frame_inds + (unsafe_inds.T * last_ind).T
                frame_inds = new_inds
            else:
                raise ValueError("Illegal out_of_bound option.")
            frame_inds = frame_inds.squeeze().astype(int).tolist()
            frames = []
            for i in frame_inds:
                try:
                    image = Image.open(io.BytesIO(open(image_paths[i], "rb").read())).convert("RGB")
                except:
                    image = Image.new("RGB", (256, 256), (255, 255, 255))
                    print("error_image_path", image_paths[i])
                    # raise Exception
                # image = Image.new("RGB", (256, 256), (255, 255, 255))
                # TODO: keep image transformation same in temporal order
                image = self.transform(image)
                frames.append(image)
        else:
            # 视频模态缺失
            image = Image.new("RGB", (256, 256), (255, 255, 255))
            image = self.transform(image)
            frames = [image] * self.num_clips
        return torch.stack(frames, dim=0)

    def _get_train_clips(self, num_frames):
        """Get clip offsets in train mode.
        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.
        Args:
            num_frames (int): Total number of frame in the video.
        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval

        if self.keep_tail_frames:
            avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)
            if num_frames > ori_clip_len - 1:
                base_offsets = np.arange(self.num_clips) * avg_interval
                clip_offsets = (base_offsets + np.random.uniform(0, avg_interval, self.num_clips)).astype(int)
            else:
                clip_offsets = np.zeros((self.num_clips,), dtype=int)
        else:
            avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

            if avg_interval > 0:
                base_offsets = np.arange(self.num_clips) * avg_interval
                clip_offsets = base_offsets + np.random.randint(avg_interval, size=self.num_clips)
            elif num_frames > max(self.num_clips, ori_clip_len):
                clip_offsets = np.sort(np.random.randint(num_frames - ori_clip_len + 1, size=self.num_clips))
            elif avg_interval == 0:
                ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
                clip_offsets = np.around(np.arange(self.num_clips) * ratio)
            else:
                clip_offsets = np.zeros((self.num_clips,), dtype=int)

        return clip_offsets

    def _get_test_clips(self, num_frames):
        """Get clip offsets in test mode.
        Calculate the average interval for selected frames, and shift them
        fixedly by avg_interval/2. If set twice_sample True, it will sample
        frames together without fixed shift. If the total number of frames is
        not enough, it will return all zero indices.
        Args:
            num_frames (int): Total number of frame in the video.
        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)
        if num_frames > ori_clip_len - 1:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = (base_offsets + avg_interval / 2.0).astype(int)
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=int)
        return clip_offsets

    def _sample_clips(self, num_frames):
        """Choose clip offsets for the video in a given mode.
        Args:
            num_frames (int): Total number of frame in the video.
        Returns:
            np.ndarray: Sampled frame indices.
        """
        if self.test_mode:
            clip_offsets = self._get_test_clips(num_frames)
        else:
            clip_offsets = self._get_train_clips(num_frames)
        return clip_offsets


# @customized_processor(verbose=True)
class AuthenticProcessor:
    def __init__(
        self,
        test_mode,
        vocab_file="hdfs://haruna/home/byte_ecom_govern/user/liuyuhang/pretrain/zh_old_cut_145607.vocab",
        # vocab_file = '/mnt/bn/ecom-govern-maxiangqian/qingxuan/EasyGuard/easyguard/appzoo/authentic_modeling/chinese_roberta_wwm_ext_pytorch/vocab.txt',
        text_max_len=128,
        frame_len=8,
        text_keys=("asr_text", "ocr_text"),
        # text_keys=('asr_text', 'video_ocr_high_value', 'video_ocr_is_caption',
        #           'video_ocr_is_title'),
        frame_key="frame_file_paths",
        label_keys=("label",),
    ):
        self.text_process = TextProcessor(vocab_file=vocab_file, max_len=text_max_len)
        self.frame_process = VideoFrameProcess(test_mode, num_clips=frame_len)
        self.text_keys = text_keys
        self.frame_key = frame_key
        self.label_keys = label_keys

    def transform(self, data_dict: dict):
        # parse text input
        all_token_ids = []
        all_segment_ids = []
        all_attn_mask = []
        for tk in self.text_keys:
            token_ids, masks, segment_ids = self.text_process(data_dict[tk])
            all_token_ids.append(token_ids)
            all_segment_ids.append(segment_ids)
            all_attn_mask.append(masks)
        # parse frame input
        frame_paths = data_dict[self.frame_key]
        try:
            item_id = frame_paths[0].split("/")[-2]
        except:
            item_id = None
        frames = self.frame_process(frame_paths)
        # print('data shape: token_ids({})'.format(len(token_ids)))
        # print('data shape: token_ids({}), frames({})'.format(len(token_ids), frames.shape))
        ret = {
            "token_ids": all_token_ids,
            "segment_ids": all_segment_ids,
            "attn_mask": all_attn_mask,
            "frames": frames,
            "item_id": item_id,
        }
        # parse label
        labels = {}
        for lk in self.label_keys:
            labels[lk] = data_dict[lk]
        ret.update(labels)
        return ret

    def batch_transform(self, data):
        keys = list(data[0].keys())
        batch = {k: [] for k in keys}
        for i, ibatch in enumerate(data):
            for k in keys:
                batch[k].append(ibatch[k])

        for k in keys:
            if k != "item_id":
                if isinstance(batch[k][0], torch.Tensor):
                    batch[k] = torch.stack(batch[k], dim=0)
                elif isinstance(batch[k][0], np.ndarray):
                    batch[k] = torch.from_numpy(np.stack(batch[k], axis=0))
                else:
                    batch[k] = torch.tensor(batch[k])
        return batch


class AuthenticDataModule(CruiseDataModule):
    def __init__(
        self,
        train_files: str = None,
        val_files: str = None,
        train_batch_size: int = 64,
        val_batch_size: int = 32,
        num_workers: int = 24,
        max_len: int = 256,
        frame_len: int = 8,
        vocab_file: str = None,
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
            processors=AuthenticProcessor(
                False,
                text_max_len=self.hparams.max_len,
                frame_len=self.hparams.frame_len,
                vocab_file=self.hparams.vocab_file,
            ),
            decode_fn_list=[AuthenticDataDecode()],
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
            processors=AuthenticProcessor(
                True,
                text_max_len=self.hparams.max_len,
                frame_len=self.hparams.frame_len,
                vocab_file=self.hparams.vocab_file,
            ),
            decode_fn_list=[AuthenticDataDecode()],
        )

    def predict_dataloader(self, data_source=None, batch_size=32, num_workers=8):
        return create_cruise_loader(
            data_sources=data_source,
            data_types="kv",
            batch_sizes=batch_size,
            num_workers=num_workers,
            num_readers=4,
            shuffle=False,
            processors=AuthenticProcessor(
                True,
                text_max_len=self.hparams.max_len,
                frame_len=self.hparams.frame_len,
                vocab_file=self.hparams.vocab_file,
            ),
            decode_fn_list=[AuthenticDataDecode()],
            drop_last=False,
        )
