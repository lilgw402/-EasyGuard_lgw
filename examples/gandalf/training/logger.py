import re
import warnings

from cruise.trainer.logger import TensorBoardLogger
from cruise.trainer.logger.tracking import TrackingLogger
from cruise.trainer.logger.trainer_default import create_default_cruise_loggers, create_default_cruise_loggers_by_str
from cruise.utilities.rank_zero import rank_zero_info


def init_loggers(train_kwargs):
    log_path = "./events_log"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cruise_loggers = [
            "console",
            TensorBoardLogger(
                save_dir=log_path,
                hdfs_dir=None,
                name="",
                flush_logs_every_n_steps=max(1, 100 // train_kwargs.summary_interval),
                version="",
                ignore_keys=["__elapsed__"],
            ),
        ]
    try:
        # add tracking logger
        if train_kwargs.get("tracking_project_name", ""):
            tracking_project_name = train_kwargs["tracking_project_name"]
            if "/" in tracking_project_name:
                project, name = tracking_project_name.rsplit("/", 1)
                project = project.replace("/", "_")
                # remove special chars
                name = "_".join(re.findall(r"[a-zA-Z0-9\u4E00-\u9FA5-_./@]{1,128}", name))
            else:
                project = tracking_project_name
                name = ""
            cruise_loggers.append(
                TrackingLogger(
                    project=project,
                    name=name,
                    config={"trainer": train_kwargs},
                    version="",
                    ignore_keys=["__elapsed__"],
                    allow_keys=["training/grad_norm"],
                )
            )
            rank_zero_info("Tracking enabled with name: {}".format(tracking_project_name))
    except ImportError:
        rank_zero_info("Tracking not enabled")
    return cruise_loggers
