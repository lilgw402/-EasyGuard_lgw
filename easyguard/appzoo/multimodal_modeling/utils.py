# -*- coding: utf-8 -*-

import collections
import io
import unicodedata
import urllib

import torch
import torchvision.transforms as transforms
from cruise.utilities.hdfs_io import hopen
from PIL import Image

from .downloads import download_url_with_exception, further_real_url, get_original_urls


class ImageProcess(object):
    def __init__(self, mode="train"):
        self.transform = self.get_transform(mode)

    def get_transform(self, mode: str = "train"):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if mode == "train":
            com_transforms = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.01, 0.2)),
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

    """
    Choose one url from url list
    """

    def __call__(self, urls):
        if isinstance(urls, str):
            urls = [urls]
        urls = get_original_urls(urls)

        try:
            image_str = b""
            for url in urls:
                image_str = download_url_with_exception(url, timeout=3)
                if image_str != b"" and image_str != "":
                    break
                else:
                    url = further_real_url(url)
                    image_str = download_url_with_exception(url, timeout=3)
                    if image_str != b"" and image_str != "":
                        break
            image = Image.open(io.BytesIO(image_str)).convert("RGB")
        except:
            image_str = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\xff\xdb\x00C\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\xff\xc0\x00\x11\x08\x00\x02\x00\x02\x03\x01"\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x15\x00\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\n\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xc4\x00\x14\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xc4\x00\x14\x11\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00?\x00\xbf\x80\x01\xff\xd9'
            image = Image.open(io.BytesIO(image_str)).convert("RGB")

        image = self.transform(image)  # [3, 224, 224]

        return image


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    if vocab_file.startswith("hdfs://"):
        with hopen(vocab_file, "r") as reader:
            accessor = io.BytesIO(reader.read())
            while True:
                token = accessor.readline()
                token = token.decode("utf-8")  # 要解码使得数据接口类型一致
                if not token:
                    break
                token = token.strip()
                vocab[token] = index
                index += 1
            del accessor
            return vocab
    else:
        with open(vocab_file, "r", encoding="utf-8") as reader:
            while True:
                token = reader.readline()
                if not token:
                    break
                token = token.strip()
                vocab[token] = index
                index += 1
            return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(
        self,
        do_lower_case=True,
        tokenize_emoji=False,
        never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"),
    ):
        """Constructs a BasicTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case
        self.never_split = never_split
        self.tokenize_emoji = tokenize_emoji

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = self._clean_text(text)
        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        # text = self._tokenize_chinese_chars(text)
        if self.tokenize_emoji:
            text = self._tokenize_emoji_chars(text)  # emoji 两边加空格
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in self.never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        if text in self.never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_emoji_chars(self, text):
        """Adds whitespace around any emoji character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_emoji_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_emoji_char(self, char_code):
        """
        https://gist.github.com/msenol86/44082269be46aa446ccda9d02202e523
        """
        if not char_code:
            return False
        range_min = ord("\U0001F300")  # 127744
        range_max = ord("\U0001FAD6")  # 129750
        range_min_2 = 126980  # 0x1f004
        range_max_2 = 127569  # 0x1f251
        range_min_3 = 169  # 0xa9
        range_max_3 = 174
        range_min_4 = 8205
        range_max_4 = 12953
        if range_min <= char_code <= range_max:
            # or range_min_2 <= char_code <= range_max_2 or range_min_3 <= char_code <= range_max_3 or range_min_4 <= char_code <= range_max_4:
            return True
        elif range_min_2 <= char_code <= range_max_2:
            return True
        elif range_min_3 <= char_code <= range_max_3:
            return True
        elif range_min_4 <= char_code <= range_max_4:
            return True
        else:
            return False

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(
        self,
        vocab,
        greedy_sharp=True,
        unk_token="[UNK]",
        # max_input_chars_per_word=100,
        max_input_chars_per_word=512,
    ):
        self.vocab = vocab
        self.greedy_sharp = greedy_sharp
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    # **注意** greedy_sharp 这里的逻辑和原始的bert的tokenizer不太一样，
                    # 原始的tokenizer如果遇到没有##x，就会跳过，这里还会再看看 x是不是在词表里
                    # greedy_sharp的版本一是兼容之前有一些无##的词表，二是更大程度的减少[UNK]
                    if start > 0:
                        slash_substr = "##" + substr
                        if slash_substr in self.vocab:
                            cur_substr = slash_substr
                            break
                        elif self.greedy_sharp and substr in self.vocab:
                            cur_substr = substr
                            break
                    elif substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


class BertTokenizer(object):
    """Runs end-to-end tokenization: punctuation splitting + wordpiece"""

    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        tokenize_emoji=False,
        greedy_sharp=True,
        max_len=None,
        never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"),
    ):
        """
        vocab_file: 词表文件
        do_lower_case: 是否小写
        tokenzie_emoji: 是否对emoji两端加空格，这样emoji会单做单独token
        greedy_sharp: 是否贪心的匹配 无##的词。
            在bpe的时候，对被切开的词的后半部分，必须得是 "##x" 的形式，这样会增加UNK的比例。
            默认google的tokenizer是greedy_sharp=False 的形式。
            如果greedy_sharp 是true，则会先看 "##x" 是在词表里，如果不在，会看 "x" 是否在词表里。
        """
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.basic_tokenizer = BasicTokenizer(
            do_lower_case=do_lower_case,
            tokenize_emoji=tokenize_emoji,
            never_split=never_split,
        )
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, greedy_sharp=greedy_sharp)
        self.max_len = max_len if max_len is not None else int(1e12)

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def tokenize_with_ww(self, text):
        """保留 word info 的 tokenize"""
        split_tokens = []
        word_infos = []
        for i, token in enumerate(self.basic_tokenizer.tokenize(text)):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
                word_infos.append(i)
        return split_tokens, word_infos

    def recover(self, tokens):
        """
        将 token 恢复成原本的term。注意只能做到term级别，因为空格信息不知道
        """
        return "".join([t.replace("##", "") for t in tokens])

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        if len(ids) > self.max_len:
            raise ValueError(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(len(ids), self.max_len)
            )
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens


class TextProcess(object):
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

    def __call__(self, product_name: str, image_ocr: str = ""):
        product_name_tokens = self.tokenizer.tokenize(product_name)[: self.max_len // 2]
        image_ocr_tokens = self.tokenizer.tokenize(image_ocr)
        image_ocr_tokens = image_ocr_tokens[: self.max_len - 3 - len(product_name_tokens)]

        text_masks = []
        text_segment_ids = []
        tokens = ["[CLS]"] + product_name_tokens + ["[SEP]"] + image_ocr_tokens + ["[SEP]"]
        text_masks.extend([1] * len(tokens))
        text_segment_ids.extend([0] + [0] * len(product_name_tokens) + [0] + [1] * len(image_ocr_tokens) + [1])

        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        text_masks.extend([0] * (self.max_len - len(token_ids)))  # Pad
        text_segment_ids.extend([0] * (self.max_len - len(token_ids)))  # Pad

        token_ids = token_ids + [self.PAD] * (self.max_len - len(token_ids))  # 填充至最大长度

        return token_ids, text_masks, text_segment_ids
