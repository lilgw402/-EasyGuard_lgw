import os
import random
import re
import subprocess
import time
from contextlib import contextmanager

from utils.driver import HADOOP_BIN, get_logger
from utils.util import async_run


def push_files(local_path, hdfs_path):
    try:
        if not check_hdfs_exist(hdfs_path):
            cmd = f"{HADOOP_BIN} fs -mkdir -p {hdfs_path}"
            os.system(cmd)
        cmd = f"{HADOOP_BIN} fs -put -f {local_path} {hdfs_path}"
        get_logger().info(f"run command: {cmd}")
        os.system(cmd)
        return True
    except Exception as e:
        get_logger().error(e)
        return False


def substitute_hdfs_prefix(input_dir):
    input_dir = input_dir.replace("hdfs:///user", "hdfs://haruna/user")
    return input_dir


def check_file_exist(input_dir):
    os.makedirs(input_dir, exist_ok=True)
    if not check_hdfs_exist(input_dir):
        hmkdir(input_dir)


def check_hdfs_exist(path):
    cmd = "%s fs -test -e %s" % (HADOOP_BIN, path)
    ret = os.system(cmd)
    if ret != 0:
        return False
    return True


def hmkdir(directory):
    cmd = f"{HADOOP_BIN} fs -mkdir -p " + directory
    get_logger().info(f"run command: {cmd}")
    return os.system(cmd)


def hfetch_file_list(data_dir, recursive=False):
    recur = "-R" if recursive else ""
    args = f"{HADOOP_BIN} fs -ls {recur} " + data_dir
    time1 = time.time()
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    s_output, s_err = proc.communicate()
    if s_output:
        lines = s_output.splitlines()
        lines = [line.decode("utf-8") for line in lines]
        get_logger().info(f"{data_dir} parsing time used: {round(time.time() - time1, 3)}s")
        return [line.split(" ")[-1] for line in lines]
    else:
        return []


def hfetch_file_size(data_dir, recursive=False):
    recur = "-R" if recursive else ""
    args = f"{HADOOP_BIN} fs -ls {recur} " + data_dir + " | awk '{print $5}'"
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    s_output, s_err = proc.communicate()
    file_list = [s.decode("UTF-8") for s in s_output.split()]
    return file_list


def filter_file_list(file_path_list, filename_pattern=""):
    if filename_pattern:
        patt = re.compile(filename_pattern, re.I)
        return [f for f in file_path_list if patt.findall(f.rsplit("/")[-1])]
    else:
        return [f for f in file_path_list]


def safe_int(number, default=0):
    try:
        digit = int(number)
    except Exception as e:  # noqa: F841
        digit = default
    return digit


@contextmanager
def hdfs_open(hdfs_path: str, mode: str = "r"):
    """open hdfs file using with context"""
    if mode.startswith("r"):
        pipe = subprocess.Popen(
            "{} fs -text {}".format(HADOOP_BIN, hdfs_path),
            shell=True,
            stdout=subprocess.PIPE,
        )
        yield pipe.stdout
        pipe.stdout.close()  # type: ignore
        pipe.wait()
        return
    if mode == "wa" or mode == "a":
        pipe = subprocess.Popen(
            "{} fs -appendToFile - {}".format(HADOOP_BIN, hdfs_path),
            shell=True,
            stdin=subprocess.PIPE,
        )
        yield pipe.stdin
        pipe.stdin.close()  # type: ignore
        pipe.wait()
        return
    if mode.startswith("w"):
        pipe = subprocess.Popen(
            "{} fs -put -f - {}".format(HADOOP_BIN, hdfs_path),
            shell=True,
            stdin=subprocess.PIPE,
        )
        yield pipe.stdin
        pipe.stdin.close()  # type: ignore
        pipe.wait()
        return
    raise RuntimeError("unsupported io mode: {}".format(mode))


def hopen(hdfs_path: str, mode: str = "r"):
    is_hdfs = hdfs_path.startswith("hdfs")
    if is_hdfs:
        return hdfs_open(hdfs_path, mode)
    else:
        return open(hdfs_path, mode)


def scan_hdir(hdfs_dir, suffix=".parquet", shuffle=False):
    """
    find only files recursively with specified suffix
    """
    result = hfetch_file_list(hdfs_dir, recursive=True)
    file_list = [fname for fname in result if fname.endswith(suffix)]
    if shuffle:
        random.shuffle(file_list)
    else:
        file_list.sort()
    return file_list


def scan_hdfs_dir(
    hdfs_dir,
    folder,
    file_pattern,
    shuffle=False,
    stick_folder_file=False,
    min_size=None,
):
    """
    scan files from hdfs, find only files which match file_pattern
    1. filter all zero size
    2. filter all files without f_keyword
    3. support hdfs_dir with regex
    """
    # speed up
    if stick_folder_file:
        pattern_list = [os.path.join(hdfs_dir, f) + file_pattern for f in folder.split("|")]
    else:
        pattern_list = [os.path.join(hdfs_dir, f, file_pattern) for f in folder.split("|")]

    file_list = async_run(hfetch_file_list, pattern_list, pool_size=8, use_thread=True)

    # flatten
    flatten_file_list = []
    for f in file_list:
        flatten_file_list.extend(f)
    get_logger().info(f"Fetch {len(flatten_file_list)} files")

    if min_size is not None:
        org_len = len(flatten_file_list)
        file_size = async_run(hfetch_file_size, pattern_list, pool_size=8, use_thread=True)
        flatten_file_size = [int(x) for x in flatten_list(file_size)]  # noqa: F821
        flatten_file_list = [flatten_file_list[i] for i in range(org_len) if flatten_file_size[i] > min_size]
        get_logger().info(
            f"MinSize Filter: filter out {org_len - len(flatten_file_list)} files which less than {min_size}"
        )
    flatten_file_list = list(set(flatten_file_list))
    # post process
    if shuffle:
        random.shuffle(flatten_file_list)
    else:
        flatten_file_list.sort()
    return flatten_file_list


def scan_local_dir(folder_path, folder, filename_pattern, shuffle=False):
    """scan files on local path"""
    import glob

    file_list = glob.glob(os.path.join(folder_path, folder, filename_pattern))
    if shuffle:
        random.shuffle(file_list)
    return file_list
