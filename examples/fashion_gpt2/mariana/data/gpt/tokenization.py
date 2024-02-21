import base64
import collections
import copy
import json
import logging
import os
import re
import sys
import tempfile
import unicodedata
from abc import ABCMeta, abstractmethod
from builtins import len

import torch

try:
    import sentencepiece as spm
except ImportError:
    logging.warning("Failed to import sentencepiece")


__all__ = ["CasterTokenizer", "caster_bert"]


class BaseCaster(metaclass=ABCMeta):
    """Base class for caster, only do logging in this class."""

    def __init__(self):
        self.name = self.__class__.__name__
        logging.debug(
            "{}: {}".format(
                self.name,
                json.dumps(
                    self.__dict__,
                    sort_keys=True,
                    indent=2,
                    default=lambda _: "<not serializable>",
                ),
            )
        )

    @abstractmethod
    def do_cast(self, batched_input):
        """Cast input based on config.

        Args:
          batched_input (list or tf.Tensor): input should be a batch of elements.

        Returns:

        """
        raise NotImplementedError


def get_emoji_regex_pattern(mode="py"):
    """Return emoji regex pattern for different backends.

    Args:
      mode (str): py, tensorflow or pytorch.

    Returns:
      str: regex pattern of emoji.
    """
    _common_emoji_codes = [
        (0x0000200D,),
        (0x0000FE0F,),
        (0x00002600, 0x000027BF),
        (0x0001F300, 0x0001F64F),
        (0x0001F680, 0x0001F6FF),
    ]

    _full_emoji_codes = [
        (0x0000200D,),
        (0x0000FE0F,),
        (0x00002600, 0x000027BF),
        (0x0001F300, 0x0001F64F),
        (0x0001F680, 0x0001FAFF),
    ]

    if mode == "tensorflow":
        single_codes = [
            r"\x{{{}}}".format(hex(emoji_tuple[0])[2:]) for emoji_tuple in _full_emoji_codes if len(emoji_tuple) == 1
        ]

        codes_range = [
            r"[\x{{{}}}-\x{{{}}}]".format(hex(emoji_tuple[0])[2:], hex(emoji_tuple[1])[2:])
            for emoji_tuple in _full_emoji_codes
            if len(emoji_tuple) == 2
        ]
        return "|".join(single_codes + codes_range)

    single_codes = [chr(emoji_tuple[0]) for emoji_tuple in _full_emoji_codes if len(emoji_tuple) == 1]
    codes_range = [
        "[{}-{}]".format(chr(emoji_tuple[0]), chr(emoji_tuple[1]))
        for emoji_tuple in _full_emoji_codes
        if len(emoji_tuple) == 2
    ]
    return "|".join(single_codes + codes_range)


class EmojiPadding(BaseCaster):
    """Pad a space around the emoji."""

    def __init__(self):
        self.regex_pattern = re.compile("({})".format(get_emoji_regex_pattern("purepy")))
        logging.debug("EmojiPadding regex_pattern: {}.".format(self.regex_pattern))
        super(EmojiPadding, self).__init__()

    def _pad_emoji(self, text):
        return self.regex_pattern.sub(r" \1 ", text)

    def do_cast(self, batched_input):
        return list(map(self._pad_emoji, batched_input))


class PunctuationPadding(BaseCaster):
    """Pad a space around the punctuation."""

    @staticmethod
    def _is_punctuation(char):
        """Checks whether `chars` is a punctuation character."""
        char_ord = ord(char)
        # We treat all non-letter/number ASCII as punctuation.
        # Characters such as "^", "$", and "`" are not in the Unicode
        # Punctuation class but we treat them as punctuation anyways, for
        # consistency.
        if (33 <= char_ord <= 47) or (58 <= char_ord <= 64) or (91 <= char_ord <= 96) or (123 <= char_ord <= 126):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False

    def _pad_punc(self, text):
        new_text = []
        for char in text:
            if self._is_punctuation(char):
                new_text.extend([" ", char, " "])
            else:
                new_text.append(char)
        return "".join(new_text)

    def do_cast(self, batched_input):
        return list(map(self._pad_punc, batched_input))


class SpaceSplitter(BaseCaster):
    """Split each string in the batched input by space."""

    @staticmethod
    def _split_by_space(text):
        return text.split()

    def do_cast(self, batched_input):
        """Split any whitespace.

        Args:
          batched_input (list): list of str to split by space.

        Returns:
          list: return list of list of str, [[str1, str2, ...], [str1, str2], ...]
        """
        return list(map(self._split_by_space, batched_input))


class CharSplitter(BaseCaster):
    """Split each string in the batched input into characters."""

    @staticmethod
    def _split_all_chars(text):
        return list(text)

    def do_cast(self, batched_input):
        """Split string to chars.

        Args:
          batched_input (list): list of str to split to chars.

        Returns:
          list: return list of list of str,
            [[char1, char2, ...], [char1, char2], ...]
        """
        return list(map(self._split_all_chars, batched_input))


class TextCleaner(BaseCaster):
    """Clean control chars and normalize whitespace."""

    @staticmethod
    def _is_whitespace(char):
        """Checks whether `chars` is a whitespace character."""
        # \t, \n, and \r are technically control characters but we treat them
        # as whitespace since they are generally considered as such.
        if char in {" ", "\t", "\n", "\r"}:
            return True
        cat = unicodedata.category(char)
        if cat == "Zs":
            return True
        return False

    @staticmethod
    def _is_control(char):
        """Checks whether `chars` is a control character."""
        # These are technically control characters but we count them as whitespace
        # characters.
        if char in {"\t", "\n", "\r"}:
            return False
        cat = unicodedata.category(char)
        if cat in {"Cc", "Cf", "Mn"}:
            return True
        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            char_ord = ord(char)
            if self._is_whitespace(char):
                output.append(" ")
            elif char_ord == 0 or char_ord == 0xFFFD or self._is_control(char):
                continue
            else:
                output.append(char)
        return "".join(output)

    def do_cast(self, batched_input):
        return list(map(self._clean_text, batched_input))


class CaseNormalizer(BaseCaster):
    """Normalize case."""

    def __init__(self, do_lowercase=False, unicode_norm=None):
        if unicode_norm and unicode_norm.upper() not in {
            "NFD",
            "NFC",
            "NFKD",
            "NFKC",
        }:
            logging.warning("unicode norm {} is invalid.".format(unicode_norm))
            unicode_norm = None

        self.unicode_norm = unicode_norm.upper() if unicode_norm else None
        self.do_lowercase = do_lowercase
        super(CaseNormalizer, self).__init__()

    def _normalize_case(self, text):
        if self.do_lowercase:
            text = text.lower()

        if self.unicode_norm:
            text = unicodedata.normalize(self.unicode_norm, text)
        return text

    def do_cast(self, batched_input):
        return list(map(self._normalize_case, batched_input))


class Libcut(BaseCaster):
    """Break words into subwords."""

    def __init__(self, libcut_config, cut_mode="search"):
        """

        Args:
          libcut_config (str or dict): str of json config.
          cut_mode (str):

        """
        try:
            from text_cutter.cut import Cutter
        except ImportError:
            logging.error(
                "Unable to import `Cutter` from `text_cutter.cut`,"
                "libcut_py may not been installed, try `pip3 install https://luban-source.byted.org/repository/scm/search.nlp.libcut_py_2.3.0.48.tar.gz --user --upgrade --force-reinstall`"
            )
            raise

        # TODO: support more flexiable configuration
        self.libcut_config = libcut_config
        self.cut_mode = cut_mode
        # TODO: support old cut
        if not self.cut_mode.startswith("libcut_v2"):
            raise ValueError("Caster with purepy backend cannot run " "when cut_mode is not libcut_v2*")
        self.cut_type = "CRF_LARGE"
        if self.cut_mode == "libcut_v2_fine":
            self.cut_level = "FINE"
        elif self.cut_mode == "libcut_v2_coar":
            self.cut_level = "COARSE"
        else:
            self.cut_level = "DEFAULT"
        self.cutter = Cutter(self.cut_type, resource_path=self.libcut_config)
        super(Libcut, self).__init__()

    # pylint: disable=unused-variable
    def do_cast(self, batched_input):
        """Cut each string in the batched input by libcut.

        Args:
          batched_input (tf.Tensor): a batch of string to split by space.

        Returns:
          tf.RaggedTensor: a ragged tensor with shape [batch_size, total_tokens].

        """
        # TODO: support concat cut
        batched_output = []
        for input_text in batched_input:
            fine_res_tokens = self.cutter.cut(input_text, cut_level=self.cut_level)
            batched_output.append([token.strip() for token in fine_res_tokens])
        return batched_output


class SentencePieceBPE(BaseCaster):
    def __init__(
        self,
        vocab,
        model_str_b64=None,
        max_tokens_per_input=0,
        max_input_chars_per_word=1024,
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
        glossaries=None,
        **kwargs,
    ):
        """Initializes.

        Args:
            model_str_b64: model file content
            glossaries: special tokens
        """
        self.vocab = vocab
        tmpf = tempfile.NamedTemporaryFile(delete=False)
        with open(tmpf.name, "wb") as fw:
            fw.write(base64.b64decode(model_str_b64))
        self._sp = spm.SentencePieceProcessor()
        status = self._sp.Load(tmpf.name)
        os.remove(tmpf.name)
        self.max_tokens_per_input = max_tokens_per_input
        self.max_input_chars_per_word = max_input_chars_per_word
        self._bos_token, self._eos_token, self._unk_token, self._pad_token = (
            bos_token,
            eos_token,
            unk_token,
            pad_token,
        )
        self._space_id = -1
        for i in range(len(self._sp)):
            if self._sp.id_to_piece(i) == "▁":
                self._space_id = i
                break
        assert self._space_id >= 0, "SentencePiece Vocab Must include space '▁'"

        super(SentencePieceBPE, self).__init__()

    # pylint: disable=too-many-branches
    def _tokenize(self, token_list):
        """Tokenize a piece of text into its word pieces. Spaces ' ' are replaced with '▁'
        For example:
          input = "你好 world"
          output = ["你", "好", "▁wor", "ld"]

        Args:
          token_list: A single token or whitespace separated tokens.
        Returns:
          list: A list of word piece subtokens.
        """

        output_tokens = []
        for token in token_list:
            if self.max_tokens_per_input > 0 and len(output_tokens) > self.max_tokens_per_input:
                break
            if len(token) > self.max_input_chars_per_word:
                output_tokens.append(self._unk_token)
                continue
            subtokens = self._sp.encode_as_pieces(token)

            for i, subtoken in enumerate(subtokens):
                if subtoken not in self.vocab:
                    if self._unk_token == "[HASH]":
                        subtokens[i] = subtoken[0]
                    else:
                        subtokens[i] = self._unk_token
            output_tokens.extend(subtokens)
        if self.max_tokens_per_input > 0:
            return output_tokens[: self.max_tokens_per_input]
        return output_tokens

    def do_cast(self, batched_input):
        """

        Args:
          batched_input (list): list of list tokens.

        Returns:
          list: return list of list of subtokens.
        """
        return list(map(self._tokenize, batched_input))


class WordpieceTokenizer(BaseCaster):
    """Break words into subwords."""

    def __init__(
        self,
        vocab,
        unk_token="[UNK]",
        wordpiece_type="bert",
        max_input_chars_per_word=200,
        use_bpe=True,
        max_tokens_per_input=-1,
        **kwargs,
    ):
        """Word piece Tokenizer.

        Args:
          vocab (set or dict): contains set of known tokens.
          unk_token (str): "[UNK]"
          wordpiece_type (str): bert or t2t.
          max_input_chars_per_word (int): ignore long words.
          use_bpe (bool): if use_bpe, prepend "##" to subtokens.
          max_tokens_per_input (int): pad or remove tokens.
        """
        self.vocab = vocab
        self.unk_token = unk_token
        self.wordpiece_type = wordpiece_type
        self.max_input_chars_per_word = max_input_chars_per_word
        self.use_bpe = use_bpe
        self.max_tokens_per_input = max_tokens_per_input
        logging.warning("WordpieceTokenizer kwargs: {} is ignored.".format(kwargs))

        super(WordpieceTokenizer, self).__init__()

    # pylint: disable=too-many-branches
    def _tokenize(self, token_list):
        """Tokenize a piece of text into its word pieces.
        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.
        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          token_list: A single token or whitespace separated tokens.
        Returns:
          list: A list of word piece subtokens.
        """

        output_tokens = []
        for token in token_list:
            if self.max_tokens_per_input > 0 and len(output_tokens) > self.max_tokens_per_input:
                break
            if self.use_bpe and self.wordpiece_type == "t2t":
                token = token.replace("\\", "\\\\").replace("_", "\\u") + "_"
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                flag = False
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        if self.use_bpe and self.wordpiece_type == "bert":
                            substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                    flag = True
                if (cur_substr is None) and flag:
                    if self.unk_token == "[HASH]":
                        # return tokenized token for later hash
                        sub_tokens.append(chars[start])
                    else:
                        sub_tokens.append(self.unk_token)
                    start = start + 1
                else:
                    sub_tokens.append(cur_substr)
                    start = end
            output_tokens.extend(sub_tokens)
        if self.max_tokens_per_input > 0:
            return output_tokens[: self.max_tokens_per_input]
        return output_tokens

    def do_cast(self, batched_input):
        """

        Args:
          batched_input (list): list of list tokens.

        Returns:
          list: return list of list of subtokens.
        """
        return list(map(self._tokenize, batched_input))


class VocabLookup(BaseCaster):
    """Convert str to int ids."""

    def __init__(self, vocab, unk_token="[UNK]", unk_id=None, num_oov_buckets=0):
        """

        Args:
          vocab (dict): str to id mapping.
          unk_token (str): "[UNK]"
          unk_id (int): the index of unk_token
          num_oov_buckets (int):
            the number of buckets for out-of-vocabulary token hash

        """
        self.vocab = vocab
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.unk_token = unk_token
        if unk_id is None:
            self.unk_id = vocab[unk_token]
        else:
            self.unk_id = unk_id
        self.num_oov_buckets = num_oov_buckets
        # if self.num_oov_buckets > 0:
        #   logging.warning(
        #     "Only fingerprint64 hashing is supported for vocab_ops under purepy backend.")
        super(VocabLookup, self).__init__()

    def _lookup(self, token_list):
        def token2idx(token):
            idx = self.vocab.get(token, self.unk_id)
            # pylint: disable=line-too-long
            # ref: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/lookup_ops.py#L946
            if idx == self.unk_id and self.num_oov_buckets > 0:
                import farmhash

                idx = farmhash.fingerprint64(token) % self.num_oov_buckets + len(self.vocab)
            return idx

        return list(map(token2idx, token_list))

    def do_cast(self, batched_input):
        """

        Args:
          batched_input (list): list of list tokens.

        Returns:
          list: return list of list of subtokens.
        """
        return list(map(self._lookup, batched_input))


_CASTERS = {
    "EmojiPadding": EmojiPadding,
    "PunctuationPadding": PunctuationPadding,
    "SpaceSplitter": SpaceSplitter,
    "CharSplitter": CharSplitter,
    "TextCleaner": TextCleaner,
    "CaseNormalizer": CaseNormalizer,
    "Libcut": Libcut,
    "VocabLookup": VocabLookup,
    "WordpieceTokenizer": WordpieceTokenizer,
    "SentencePieceBPE": SentencePieceBPE,
}


def parse_json_tokenizer_config(json_file, libcut_data_path=None):
    """Parse config according to model_name in json_file.

    Args:
      json_file (str): json model config file.
      libcut_data_path (str): Optional str.

    Returns:
      list: list of config.
    """
    with open(json_file, "r") as reader:
        text = reader.read()
        caster_configs = json.loads(text)
        if libcut_data_path is None:
            return caster_configs

        new_caster_configs = []
        # pylint: disable=too-many-nested-blocks
        for config in caster_configs:
            if config[0] == "Libcut":
                libcut_params = config[1]
                libcut_config = libcut_params["libcut_config"]
                cut_mode = libcut_params.get("cut_mode", "search")
                if cut_mode.startswith("libcut_v2"):
                    libcut_config = libcut_data_path
                else:
                    # Compatiable to old version
                    for key in libcut_config:
                        if isinstance(libcut_config[key], str):
                            # TODO:
                            val = libcut_config[key]
                            if key != "SEARCH_DATA_PATH":
                                new_val = os.path.join(
                                    libcut_data_path,
                                    "/".join(val.split("/")[-2:]),
                                )
                            else:
                                new_val = os.path.join(libcut_data_path, val.split("/")[-1])
                            logging.info("Change libcut path from {} to {}.".format(val, new_val))
                            libcut_config[key] = new_val
                libcut_params["libcut_config"] = libcut_config
                new_caster_configs.append(("Libcut", libcut_params))
            else:
                new_caster_configs.append(config)

        return new_caster_configs


class CasterTokenizer:
    """Convert text to text or ids."""

    # pylint: disable=super-init-not-called
    def __init__(self, caster_configs):
        caster_configs = copy.deepcopy(caster_configs)
        self.casters = []
        self.vocab_dict = None
        self.backend_vocab = None
        for caster_name, caster_params in caster_configs:
            if "vocab_str" in caster_params:
                # separate by \n
                vocab_str = caster_params.pop("vocab_str")
                unk_id = caster_params.get("unk_id", 4)
                num_oov_buckets = caster_params.get("num_oov_buckets", 0)
                # TODO: support multiple vocab_dict
                self.vocab_dict = {token: idx for idx, token in enumerate(vocab_str.split("\n"))}
                self.backend_vocab = self.vocab_dict
                caster_params["vocab"] = self.backend_vocab
                caster_params["num_oov_buckets"] = num_oov_buckets
                caster_params["unk_id"] = unk_id

            assert caster_name in _CASTERS, f"Invalid caster: {caster_name}"
            self.casters.append(_CASTERS.get(caster_name)(**caster_params))

        assert self.vocab_dict is not None, "Unable to parse valid vocab_str from config."

    @property
    def pad_token_id(self):
        return self.vocab_dict["[PAD]"]

    @property
    def eos_token_id(self):
        return self.vocab_dict["[EOS]"]

    @property
    def sep_token_id(self):
        return self.vocab_dict["[SEP]"]

    @classmethod
    def from_pretrained(cls, path: str, do_lower_case=None, max_len=-1, **kwargs):
        """Load from a pretrained dir, currently only supports loading from a local directory"""
        assert os.path.isdir(path), f"Invalid pretrained tokenizer directory: {path}"
        config_json = os.path.join(path, "config.json")
        assert os.path.isfile(config_json), f"Unable to find valid config.json in {path}"

        caster_configs = parse_json_tokenizer_config(config_json, libcut_data_path=path)

        for caster_config in caster_configs:
            if caster_config[0] == "CaseNormalizer" and do_lower_case is not None:
                caster_config[1]["do_lowercase"] = bool(do_lower_case)
            if caster_config[0] == "WordpieceTokenizer" and isinstance(max_len, int):
                caster_config[1]["max_tokens_per_input"] = max_len
            if caster_config[0] == "SentencePieceBPE" and isinstance(max_len, int):
                caster_config[1]["max_tokens_per_input"] = max_len

        tokenizer = CasterTokenizer(caster_configs)
        return tokenizer

    def do_cast(self, batched_input):
        unsqueeze = False
        if isinstance(batched_input, str):
            batched_input = [batched_input]
            unsqueeze = True
        output = batched_input
        for caster in self.casters:
            output = caster.do_cast(output)
        if unsqueeze:
            output = output[0]
        return output

    def do_cast_details(self, batched_input):
        """
        return all casters' results as list
        """
        if isinstance(batched_input, str):
            batched_input = [batched_input]
        outputs = []
        output = batched_input
        for caster in self.casters:
            output = caster.do_cast(output)
            outputs.append(output)
        return outputs

    def tokenize(self, text):
        unsqueeze = False
        if isinstance(text, str):
            text = [text]
            unsqueeze = True
        output = text
        for caster in self.casters:
            if isinstance(caster, VocabLookup):
                break
            output = caster.do_cast(output)
        if unsqueeze:
            output = output[0]
        return output

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        assert any(
            isinstance(caster, VocabLookup) for caster in self.casters
        ), "Unable to convert tokens to ids due to missing VocabLookup"
        unsqueeze = False
        if isinstance(tokens, (tuple, list)) and len(tokens) > 0 and isinstance(tokens[0], str):
            tokens = [tokens]
            unsqueeze = True
        for caster in self.casters:
            if isinstance(caster, VocabLookup):
                ret = caster.do_cast(tokens)
                if unsqueeze:
                    ret = ret[0]
                return ret

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        assert any(
            isinstance(caster, VocabLookup) for caster in self.casters
        ), "Unable to convert ids to tokens due to missing VocabLookup"
        for caster in self.casters:
            if isinstance(caster, VocabLookup):
                ids_to_tokens = caster.ids_to_tokens
                return [ids_to_tokens[idd] for idd in ids]

    def convert_tokens_to_string(self, tokens):
        """
        将 token 恢复成原本的term。注意只能做到term级别，因为空格信息不知道
        """
        return "".join([t.replace("##", "") for t in tokens])

    def convert_tokens_to_string_for_en(self, tokens):
        """
        将 token 恢复成原本的term。注意只能做到term级别，因为空格信息不知道
        """
        mystr = " ".join(tokens)
        mystr = mystr.replace(" ##", "")
        return mystr

    def encode(self, text):
        """The tokenized ids of the text.

        Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

        Same as doing self.convert_tokens_to_ids(self.tokenize(text)).
        """
        return self.convert_tokens_to_ids(self.tokenize(text))

    def decode(self, ids):
        """Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special tokens and clean up tokenization spaces.

        Similar to doing self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids)).
        """
        return self.convert_tokens_to_string(self.convert_ids_to_tokens(ids))

    def __call__(self, text, return_tensors=False, **kwargs):
        """A callable wrapper API to mimic transformers tokenizer output"""
        assert return_tensors is False
        input_ids = self.encode(text)
        attention_mask = [1] * len(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}
