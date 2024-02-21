"""

Dump core metrics for Rh2 eval pipeline

"""
import json
import traceback

from cruise.utilities.hdfs_io import hdfs_open, hexists
from cruise.utilities.rank_zero import rank_zero_info, rank_zero_only, rank_zero_warn

try:
    from rh2.sdk.env import get_rh2_env
except:
    pass

global_prefix = ""


@rank_zero_only
def set_metric_prefix(prefix):
    global global_prefix
    global_prefix = prefix


def hget(path) -> str:
    with hdfs_open(path, "r") as f:
        return f.read()


def _dump_metrics_to_hdfs(obj):
    global global_prefix

    rh2_env = None
    try:
        rh2_env = get_rh2_env()
    except:
        pass

    if not rh2_env:
        rank_zero_info("dump_metrics_to_hdfs: not in rh2 pipeline, skip")
        return

    filename = rh2_env.get("instance", {}).get("pipeline_run_id", rh2_env.job_run_id)
    path = f"hdfs://haruna/home/byte_aml_platform/user/eval_pipeline/{filename}.json"

    obj = {f"{global_prefix}{k}": f"{obj[k]}" for k in obj}

    # restore other metrics from the same job run
    if hexists(path):
        try:
            content = hget(path)
            old_obj = json.loads(content)
            old_obj.update(obj)
            obj = old_obj
        except:
            pass

    content = json.dumps(obj)
    with hdfs_open(path, "w") as f:
        f.write(content.encode())

    rank_zero_info(f"dump_metrics_to_hdfs: done, content={content}")


@rank_zero_only
def dump_metrics_to_hdfs(obj):
    try:
        _dump_metrics_to_hdfs(obj)
    except Exception as e:
        rank_zero_warn("dump_metrics_to_hdfs: exception")
        rank_zero_warn(str(e))
        rank_zero_warn(traceback.format_exc())
