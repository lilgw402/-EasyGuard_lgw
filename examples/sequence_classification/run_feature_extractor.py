from typing import Optional

from easyguard import AutoModel, AutoTokenizer
from easyguard.utils import logging

logger = logging.get_logger(__name__)


def main(text: Optional[str] = "hello easyguard~"):
    archive = "fashionxlm-moe-base"
    tokenizer = AutoTokenizer.from_pretrained(archive)
    inputs = tokenizer(text, return_tensors="pt", max_length=84)
    model = AutoModel.from_pretrained(archive, model_cls="sequence_model")
    ouputs = model(**inputs, language=["GB"])
    logger.info(ouputs)


if __name__ == "__main__":
    text = "hello easyguard~"
    main(text)
