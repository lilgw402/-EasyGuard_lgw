"""parse 业务/模型用途/模型结构 三层，用来自动标记产出"""
import json
import logging
import os
from datetime import datetime

import requests


def relative_basename(relative_path):
    return os.path.basename(os.path.dirname(os.path.abspath(relative_path)))


class ExpHelper:
    DEBUG_PREFIX = "dev"
    TOKEN = "19b0e0f0f7d8c0207f7b29d40d35efb3fddfdbcb"

    def __init__(self, file_path):
        self._get_trial_info()
        self._parse_model_names(file_path)

    @property
    def project_name(self):
        """Suggested tracking project name"""
        return f"lucifer_{self.model_line}_{self.model_type}_{self.model_name}"

    @property
    def hdfs_prefix(self):
        """Suggested hdfs prefix based on current trial"""
        return self._hdfs_prefix

    def _parse_model_names(self, file_path):
        full_path = os.path.abspath(os.path.expanduser(file_path))
        self.model_name = relative_basename(full_path)
        self.model_type = relative_basename(os.path.join(full_path, ".."))
        self.model_line = relative_basename(os.path.join(full_path, "..", ".."))

        # hdfs
        hdfs_prefix = os.environ.get("MARIANA_HDFS_DIR_PREFIX", None)
        if hdfs_prefix:
            assert hdfs_prefix.startswith("hdfs:")
            hdfs_prefix = os.path.join(
                hdfs_prefix,
                self.model_line,
                self.model_type,
                self.model_name,
                str(self.original_trial),
            )
        elif os.environ.get("ARNOLD_OUTPUT", ""):
            # use ARNOLD_OUTPUT
            hdfs_prefix = os.environ["ARNOLD_OUTPUT"]
        self._hdfs_prefix = hdfs_prefix

    def _get_trial_info(self):
        arnold_trial_id = os.environ.get("ARNOLD_TRIAL_ID", self.DEBUG_PREFIX)
        preemptible = False
        original_trial = arnold_trial_id
        if arnold_trial_id != self.DEBUG_PREFIX:
            try:
                headers = {
                    "Authorization": f"Token {self.TOKEN}",
                }
                response = requests.get(
                    f"http://arnold-api.byted.org/api/v2/trial/{arnold_trial_id}/",
                    headers=headers,
                )
                response.raise_for_status()
                if response.ok:
                    self.details = json.loads(response.text)
            except Exception:
                self.details = {}
            preemptible = self.details.get("preemptible", False)
            if preemptible:
                original_trial = self.details.get("ckpt_trial_id", arnold_trial_id)
                if not original_trial:
                    original_trial = arnold_trial_id

        if not preemptible and arnold_trial_id == self.DEBUG_PREFIX:
            now = datetime.now()
            date_str = now.strftime("%Y%m%d")
            original_trial = f"{self.DEBUG_PREFIX}_{date_str}"

        self.arnold_trial_id = arnold_trial_id
        self.original_trial = original_trial or "unknown_trial"
        self.preemptible = preemptible

    def report_trial_info(self, rank=None, world_size=None):
        if rank is None or world_size is None:
            from cruise.utilities.distributed import DIST_ENV

            rank = DIST_ENV.rank
            world_size = DIST_ENV.world_size
        if self.preemptible:
            logging.info(
                f"[Rank {rank}/{world_size}]In preemptible trial: {self.arnold_trial_id},"
                " original trial: {self.original_trial}"
            )
        elif self.arnold_trial_id != self.DEBUG_PREFIX:
            if self.details.get("debug", ""):
                logging.info(
                    f"[Rank {rank}/{world_size}]In arnold trial: {self.original_trial},"
                    " debug mode: {self.details.get('debug')}"
                )
            else:
                logging.info(f"[Rank {rank}/{world_size}]In normal trial: {self.original_trial}")
        else:
            logging.info("[Rank {rank}/{world_size}]In dev-machine/workspace trial where no arnold env detected.")
