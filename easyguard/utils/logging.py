import logging
import logging.config
from typing import Optional

from .config import LOGGER_CONFIG

logging.config.dictConfig(config=LOGGER_CONFIG)


# TODO (junwei.Dong): 可以扩展自定义filter机制，过滤hf的提示【适配hf的logger，warning等级】

log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

_default_log_level = logging.INFO


def _get_library_name() -> str:
    return __name__.split(".")[0]


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a logger with the specified name.

    This function is not supposed to be directly accessed unless you are writing a custom transformers module.
    """

    if name is None:
        name = _get_library_name()

    return logging.getLogger(name)
