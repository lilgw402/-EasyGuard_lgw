from collections import OrderedDict
from typing import Any, Dict

from ...processor_utils import ProcessorBase
from .image_processing_falbert import FalBertImageProcessor
from .tokenization_falbert import FalBertTokenizer


class FalBertProcessor(ProcessorBase):
    def __init__(
        self,
        vocab_path: str,
        text_processor: Dict[str, Any],
        image_processor: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        self.tokenizer = FalBertTokenizer(vocab_file=vocab_path, **text_processor["tokenizer_config"])
        self.image_processor = FalBertImageProcessor(**image_processor)

        # preprocess
        self.text_ocr = text_processor.get("text_ocr", 256)
        self.text_asr = text_processor.get("text_asr", 256)

        self.max_len = {"text_ocr": self.text_ocr, "text_asr": self.text_asr}
        self.CLS = self.tokenizer.vocab["[CLS]"]
        self.PAD = self.tokenizer.vocab["[PAD]"]
        self.SEP = self.tokenizer.vocab["[SEP]"]
        self.MASK = self.tokenizer.vocab["[MASK]"]
        self.text_types = text_processor.get("text_types", ["text_ocr", "text_asr"])

    def text_process(self, texts):
        """preprocess for text"""
        tokens = ["[CLS]"]
        for text_type in self.text_types:
            text = texts[text_type]
            tokens += self.tokenizer.tokenize(text)[: self.max_len[text_type] - 2] + ["[SEP]"]
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return OrderedDict(token_ids=token_ids)

    def image_process(self, image, **kwargs):
        """preprocess for image"""
        return super().image_process(image, **kwargs)

    def preprocess(self, text=None, image=None, **kwds):
        """default: call the self.text_process and self.image_process to process the text and image respectively"""
        return super().preprocess(text, image, **kwds)
