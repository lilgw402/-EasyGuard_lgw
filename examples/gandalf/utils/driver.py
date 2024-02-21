# -*- coding:utf-8 -*-
# Email:    jiangxubin@bytedance.com
# Created:  2023-02-27 20:04:42
# Modified: 2023-02-27 20:04:42
import logging
import os
import sys
import warnings
from functools import wraps

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
)

_logger = logging.getLogger(__name__)


def reset_logger(custom_logger):
    global _logger
    _logger = custom_logger


def get_logger():
    return _logger


def init_env(hdfs_jdk_heap_max_size="2g"):
    """envs setup"""

    # limit hdfs jvm mem & disable hdfs verbose logging
    os.environ["LIBHDFS_OPTS"] = (
        os.getenv("LIBHDFS_OPTS", "") + " -Dhadoop.root.logger=ERROR" + f" -Xms128m -Xmx{hdfs_jdk_heap_max_size}"
    )
    os.environ["KRB5CCNAME"] = "/tmp/krb5cc"

    if IN_ARNOLD:
        return

    import subprocess

    def _shell_source(script):
        """run the command in a subshell and use the results to update the current environment."""
        pipe = subprocess.Popen(". %s; env" % script, stdout=subprocess.PIPE, shell=True)
        output = pipe.communicate()[0]
        env = dict(
            (line.split("=", 1) for line in output.decode("UTF-8").splitlines() if len(line.split("=", 1)) == 2)
        )
        os.environ.update(env)

    HADOOP_HOME = "/opt/tiger/yarn_deploy/hadoop_current"
    os.environ["HADOOP_HOME"] = HADOOP_HOME
    os.environ["HADOOP_HDFS_HOME"] = "/opt/tiger/yarn_deploy/hadoop_current"
    hadoop_env_sh_path = "{}/conf/hadoop-env.sh".format(HADOOP_HOME)
    _shell_source(hadoop_env_sh_path)
    os.environ["HADOOP_CONF_DIR"] = "{}/conf".format(HADOOP_HOME)
    os.environ["LD_LIBRARY_PATH"] = "{}/jre/lib/amd64/server:{}/lib/native:{}/lib/native/ufs:{}".format(
        os.environ["JAVA_HOME"],
        os.environ["HADOOP_HDFS_HOME"],
        os.environ["HADOOP_HDFS_HOME"],
        os.environ["LD_LIBRARY_PATH"],
    )
    os.environ["CLASSPATH"] = "$(${HADOOP_HOME}/bin/hadoop classpath --glob)"
    output = subprocess.check_output(["bash", "-c", "echo $(${HADOOP_HOME}/bin/hadoop classpath --glob)"])
    os.environ["CLASSPATH"] = output.decode("UTF-8")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def init_device(local_rank=-1):
    """
    device setup
    cuda.is_available=True:
        - 若在Arnold服务器上训练，因trial的gpu配置是自定义的，所以会默认使用全部gpu，CUDA_VISIBLE_DEVICE不起作用
        - 不在Arnold上训练，CUDA_VISIBLE_DEVICE起作用
    cuda.is_available=False or CUDA_VISIBLE_DEVICE='':
        - 使用CPU进行训练
    """

    import torch
    import torch.distributed as dist

    if not IN_ARNOLD:
        assert (
            os.getenv("CUDA_VISIBLE_DEVICES", "") != ""
        ), "please set CUDA_VISIBLE_DEVICES, like: CUDA_VISIBLE_DEVICES=0 python main.py --conf xxx --enable_train"

    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        get_logger().info(f"Found {n_gpu} GPUs")
        if local_rank != -1:
            device = torch.device("cuda", local_rank)
            torch.cuda.set_device(local_rank)
        else:
            device = torch.device("cuda")

        # dist setup
        if local_rank != -1 and not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
            DIST_CONTEXT.ddp_mode = True
            DIST_CONTEXT.world_size = dist.get_world_size()
            DIST_CONTEXT.global_rank = dist.get_rank()
            DIST_CONTEXT.local_rank = local_rank
            get_logger().info(f"DDP starts world size:{DIST_CONTEXT.world_size}")
        DIST_CONTEXT.node_rank = int(os.environ.get("ARNOLD_ID", 0))
        DIST_CONTEXT.node_num = int(os.environ.get("ARNOLD_WORKER_NUM", 1))
    else:
        device = torch.device("cpu")
        n_gpu = 0

    # set context
    DIST_CONTEXT.n_gpu = n_gpu
    DIST_CONTEXT.device = device


class DistContext:
    # common
    _n_gpu = 0
    _ddp_mode = False
    _device = None
    # process view
    _local_rank = -1
    _global_rank = 0
    _world_size = 1
    # node view
    _node_rank = 0
    _node_num = 1

    def __str__(self):
        return (
            f"[DDPMode: {self._ddp_mode}, LocalRank: {self._local_rank}, GlobalRank: {self._global_rank},"
            f" WorldSize: {self._world_size}, NodeRank: {self._node_rank}, NodeNum: {self._node_num}]"
        )

    @property
    def ddp_mode(self):
        return self._ddp_mode

    @ddp_mode.setter
    def ddp_mode(self, ddp_mode):
        self._ddp_mode = ddp_mode

    @property
    def n_gpu(self):
        return self._n_gpu

    @n_gpu.setter
    def n_gpu(self, n_gpu):
        self._n_gpu = n_gpu

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = device

    @property
    def world_size(self):
        return self._world_size

    @world_size.setter
    def world_size(self, world_size):
        self._world_size = world_size

    @property
    def local_rank(self):
        return self._local_rank

    @local_rank.setter
    def local_rank(self, local_rank):
        self._local_rank = local_rank

    @property
    def global_rank(self):
        return self._global_rank

    @global_rank.setter
    def global_rank(self, global_rank):
        self._global_rank = global_rank

    @property
    def node_rank(self):
        return self._node_rank

    @node_rank.setter
    def node_rank(self, node_rank):
        self._node_rank = node_rank

    @property
    def node_num(self):
        return self._node_num

    @node_num.setter
    def node_num(self, node_num):
        self._node_num = node_num

    @property
    def is_master(self):
        return self._global_rank in [-1, 0]

    @property
    def is_local_master(self):
        return self._local_rank in [-1, 0]

    def master_only(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self._global_rank in [-1, 0]:
                return func(*args, **kwargs)

        return wrapper

    def local_master_only(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self._local_rank in [-1, 0]:
                return func(*args, **kwargs)

        return wrapper

    def barrier(self):
        from torch import distributed

        if self._world_size < 1:
            return
        if not distributed.is_available() or not distributed.is_initialized():
            return
        if distributed.get_backend() == "nccl":
            distributed.barrier(device_ids=[self._local_rank])
        else:
            distributed.barrier()


# hdfs env配置
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRIAL_ID = os.environ.get("ARNOLD_TRIAL_ID", "")
WORKSPACE_ID = os.environ.get("ARNOLD_WORKSPACE_ID", "")
IN_ARNOLD = TRIAL_ID != "" or WORKSPACE_ID != ""
DIST_CONTEXT = DistContext()
HADOOP_DIR = "/opt/tiger/yarn_deploy/hadoop"
HADOOP_BIN = HADOOP_DIR + "/bin/hadoop"

warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning, module=r"sklearn")
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
