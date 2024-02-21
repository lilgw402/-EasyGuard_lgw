import os
from typing import Optional

import bytedenv
from cruise.utilities.logger import get_cruise_logger

from easyguard.utils.tos_helper import TOSHelper

logger = get_cruise_logger()

TMP_DIR = "~/.cache/easyguard/"


def download_model_weights_from_tos(
    model_name: str,
    tos_helper: Optional[TOSHelper] = None,
    dst_dir: str = TMP_DIR,
):
    if tos_helper is None:
        idc_name = bytedenv.get_idc_name()
        if "va" in idc_name:  # 国外的TOS
            bucket_name = "ecom-govern-easyguard-aiso"
            access_key = "YD89UTTCRK4RQP5UUNZS"
        else:  # 国内的TOS
            bucket_name = "ecom-govern-easyguard-zh"
            access_key = "SHZ0CK8T8963R1AVC3WT"
        tos_helper = TOSHelper(
            bucket=bucket_name,
            access_key=access_key,
        )
    assert isinstance(tos_helper, TOSHelper), "tos_helper should be an instance of TOSHelper"
    tos_files = list(tos_helper.list_dir(model_name))
    if len(tos_files) == 0:
        logger.error(f"{model_name} is not found on EasyGuard TOS bucket.")
        raise ValueError(f"{model_name} is not found on EasyGuard TOS bucket.")
    model_path = os.path.join(dst_dir, model_name)
    os.makedirs(model_path, exist_ok=True)
    for tos_file in tos_files:
        basename = os.path.basename(tos_file)
        if os.path.exists(os.path.join(model_path, basename)):
            logger.info(f"{basename} already exists")
            continue
        else:
            logger.info(f"{basename} is downloading...")
            tos_helper.download_model_from_tos(tos_file, os.path.join(model_path, basename))
    return model_path
