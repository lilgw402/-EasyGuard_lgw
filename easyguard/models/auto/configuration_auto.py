# Very heavily inspired by:
# https://code.byted.org/lab/titan/blob/master/titan/models/auto/configuration_auto.py

from cruise.utilities.logger import get_cruise_logger
from transformers import AutoConfig as tsfm_AutoConfig

logger = get_cruise_logger()


class AutoConfig(tsfm_AutoConfig):
    logger.debug("transformers AutoConfig is replaced with EasyGuard AutoConfig")
