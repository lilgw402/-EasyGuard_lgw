import hashlib
import io
import os
import sys
import time
from typing import Any, Dict, List, Optional, OrderedDict, Union
from xml.parsers.expat import model

import torch
from prettytable import PrettyTable

from ..modelzoo.config import MODEL_ARCHIVE_PATH_BACKUP
from . import EASYGUARD_CACHE, EASYGUARD_MODEL_CACHE, REGION_MAPPING, REMOTE_PATH_SEP, SERVER_MAPPING
from .hdfs_utils import hdfs_open, hexists, hlist_files, hmget
from .logging import get_logger
from .tos_utils import TOS
from .type_utils import typecheck
from .yaml_utils import file_read

PRINT_HELP = []
# hdfs may fail to download target file, wo we use this variable to control the number of download
RETYR_TIMES = 5
FILE_TEMP = ".temp_{}"


logger = get_logger(__name__)
tos = TOS()


class HiddenPrints:
    def __init__(self, activated=True):
        self.activated = activated
        self.original_stdout = None

    def open(self):
        sys.stdout.close()
        sys.stdout = self.original_stdout

    def close(self):
        self.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __enter__(self):
        if self.activated:
            self.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.activated:
            self.open()


def file_exist(prefix: str, choices: Union[str, set]) -> Optional[str]:
    """check

    Parameters
    ----------
    prefix : str
        _description_
    choices : Union[str, set]
        _description_

    Returns
    -------
    Optional[str]
        _description_
    """
    if isinstance(choices, str):
        path_ = os.path.join(prefix, choices)
        if os.path.exists(path_):
            return path_
    else:
        for choice in choices:
            path_ = os.path.join(prefix, choice)
            if os.path.exists(path_):
                return path_
    return None


def sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def cache_file(
    model_name_path: str,
    file_name: Optional[Union[str, set]] = None,
    remote_url: Optional[str] = None,
    model_type: Optional[str] = None,
    download_retry_number: int = RETYR_TIMES,
    if_cache: Optional[bool] = False,
    *args,
    **kwargs,
) -> str:
    # TODO (junwei.Dong): 未来支持更多方式的读取
    """支持三种方式读取:
        1.本地指定文件读取: model_name_path直接指定到具体的文件
        2.本地指定模型名字读取: model_name_path + file_name + model_type, 举例: deberta_base_6l + config.yaml + debert
        3.服务器端读取: model_name_path + file_name + remote_url + model_type, 举例: deberta_base_6l + config.yaml + hdfs://haruna/home/byte_ecom_govern/easyguard/models/fashion_deberta_ccr_order + debert

    Parameters
    ----------
    model_name_path : str
        模型名字或者指定路径
    file_name : Optional[Union[str, set]], optional
        想要获取的文件,按需指定, by default None
    remote_url: Optional[str], optional
        the remote directory of model, e.g., hdfs://haruna/home/byte_ecom_govern/easyguard/models/fashion_deberta_ccr_order
    model_type: Optional[str], optional
        the type of model, for local operators, e.g., `bert`
    Returns
    -------
    str
        the local path of the target file

    Raises
    ------
    FileExistsError
        _description_
    NotImplementedError
        _description_
    """
    if os.path.exists(model_name_path):
        return model_name_path
    elif not if_cache and not remote_url.startswith("tos"):
        model_path_remote = remote_url
        assert model_path_remote is not None, f"the argument `remote_url` doesn't exist"
        model_file_remote_list = []
        if isinstance(file_name, str):
            model_file_remote_list = [REMOTE_PATH_SEP.join([model_path_remote, file_name])]
        else:
            model_file_remote_list = list(
                map(
                    lambda x: REMOTE_PATH_SEP.join([model_path_remote, x]),
                    file_name,
                )
            )
        # diffent servers, different processes
        if model_path_remote.startswith("hdfs://"):
            model_file_path_remote = None
            for remote_path_ in model_file_remote_list:
                if hexists(remote_path_):
                    model_file_path_remote = remote_path_
                    break
        if model_file_path_remote:
            return model_file_path_remote
        raise FileExistsError(f"failed to obtain one of the remote files `{model_file_remote_list}`")
    elif model_type is not None:
        hash_ = sha256(model_name_path)
        hash_file = sha256(file_name) if isinstance(file_name, str) else sha256("-".join(sorted(file_name)))

        file_temp_ = FILE_TEMP.format(hash_file)
        model_path_local = os.path.join(EASYGUARD_MODEL_CACHE, model_type, hash_)
        # a temp file to indicate whether the download is in progress
        file_temp_path = os.path.join(model_path_local, file_temp_)
        if os.path.exists(file_temp_path):
            # logger.info(
            #     f"there is a another process which is downloading the target file {file_name}"
            # )
            while os.path.exists(file_temp_path):
                time.sleep(2)

        model_file_local = file_exist(model_path_local, file_name)
        if model_file_local:
            if model_file_local not in PRINT_HELP:
                time.sleep(5)
                while os.path.exists(file_temp_path):
                    time.sleep(2)
                logger.info(f"obtain the local file `{model_file_local}`")
            PRINT_HELP.append(model_file_local)
            return model_file_local
        else:
            # TODO (junwei.Dong): 如果本地不存在那么需要根据url去远程获取，然后放置在特定的缓存目录下, 现目前只支持hdfs
            model_path_remote = (
                ("tos:" + model_name_path.replace("-", "_")) if remote_url.startswith("tos") else remote_url
            )
            assert model_path_remote is not None, f"the argument `remote_url` doesn't exist"
            os.makedirs(model_path_local, exist_ok=True)
            model_file_remote_list = []
            if isinstance(file_name, str):
                model_file_remote_list = [REMOTE_PATH_SEP.join([model_path_remote, file_name])]
            else:
                model_file_remote_list = list(
                    map(
                        lambda x: REMOTE_PATH_SEP.join([model_path_remote, x]),
                        file_name,
                    )
                )

            try:
                # if start to download, create a temp file to indicate
                file_temp_f = open(file_temp_path, "w")
                file_temp_f.close()
                # diffent servers, different processes
                if model_path_remote.startswith("hdfs://"):
                    model_file_path_remote = None
                    for remote_path_ in model_file_remote_list:
                        if hexists(remote_path_):
                            model_file_path_remote = remote_path_
                            break
                    if model_file_path_remote:
                        if model_file_path_remote not in PRINT_HELP:
                            logger.info(
                                f"start to download `{model_file_path_remote}` to local path `{model_path_local}`"
                            )
                        retry_number = 2
                        while download_retry_number > 0:
                            hmget([model_file_path_remote], model_path_local)
                            # whether the request is successful or not, the `model_file_local` will be created, so we need to check the target file
                            model_file_path_local = os.path.join(
                                model_path_local,
                                model_file_path_remote.split(REMOTE_PATH_SEP)[-1],
                            )
                            if os.path.exists(model_file_path_local):
                                if os.path.getsize(model_file_path_local) == 0:
                                    os.remove(model_file_path_local)
                                    logger.warning(
                                        f"fail to download `{model_file_path_remote}`, please check the network or the remote file path, we are trying to download the {retry_number}th time"
                                    )
                                    download_retry_number -= 1
                                    retry_number += 1
                                else:
                                    PRINT_HELP.append(model_file_path_remote)
                                    break
                    else:
                        raise FileNotFoundError(f"`{model_file_remote_list}` can not be found in remote server")
                elif model_path_remote.startswith("tos"):
                    model_file_path_remote = None
                    for remote_path_ in model_file_remote_list:
                        if tos.exist(remote_path_[4:]):
                            model_file_path_remote = remote_path_[4:]
                            break
                    if model_file_path_remote:
                        if model_file_path_remote not in PRINT_HELP:
                            logger.info(
                                f"start to download `{model_file_path_remote}` to local path `{model_path_local}`"
                            )
                        retry_number = 2
                        while download_retry_number > 0:
                            tos.get(
                                model_file_path_remote,
                                model_path_local,
                                verbose=False,
                            )
                            # whether the request is successful or not, the `model_file_local` will be created, so we need to check the target file
                            model_file_path_local = os.path.join(
                                model_path_local,
                                model_file_path_remote.split(REMOTE_PATH_SEP)[-1],
                            )
                            if os.path.exists(model_file_path_local):
                                if os.path.getsize(model_file_path_local) == 0:
                                    os.remove(model_file_path_local)
                                    logger.warning(
                                        f"fail to download `{model_file_path_remote}`, please check the network or the remote file path, we are trying to download the {retry_number}th time"
                                    )
                                    download_retry_number -= 1
                                    retry_number += 1
                                else:
                                    PRINT_HELP.append(model_file_path_remote)
                                    break
                    else:
                        raise FileNotFoundError(f"`{model_file_remote_list}` can not be found in remote server")
            finally:
                # just delete the temp file whether the download is successful or not
                if os.path.exists(file_temp_path):
                    os.remove(file_temp_path)
            final_file_path = file_exist(model_path_local, file_name)
            if final_file_path:
                return final_file_path
            raise FileExistsError(f"failed to obtain one of the remote files `{model_file_remote_list}`")
            ...

    else:
        # TODO (junwei.Dong): 不是模型的文件该如何从远程获取并存放在缓存目录
        raise NotImplementedError(f"Just support model cache")
    ...


@typecheck(str)
def hf_name_or_path_check(
    pretrained_model_name_or_path: str,
    model_url: Optional[str],
    model_type: str,
) -> str:
    """download required files from bytedance servers, which allows hf model to be used in easyguard framework
    example:
        >>> hf_name_or_path_check('fashion-deberta-ccr-order', 'hdfs://haruna/home/byte_ecom_govern/easyguard/models/fashion_deberta_ccr_order', 'vocab.txt', 'debert')
        /root/.cache/easyguard/models/debert/6ca54229f34a5b0936d549632d9566a9a07be8ac430528306e27566964dc6a4a
    Parameters
    ----------
    pretrained_model_name_or_path : str
        archive name, e.g., fashion-deberta-ccr-order
    model_url : str
        remote url
    # file_name : Union[str, set, List[set]]
    #     default names of the required files, e.g., [VOCAB_TXT, TOKENIZER_CONFIG_NAMES] or VOCAB_TXT
    model_type : str
        model architecture, e.g., debert

    Returns
    -------
    str
        the local directory which contains the downloaded files

    Raises
    ------
    ValueError
        _description_
    """
    if not model_url:
        return pretrained_model_name_or_path
    else:
        if model_url.startswith("tos"):
            file_list = tos.ls(pretrained_model_name_or_path.replace("-", "_"), if_log=False)
            if file_list:
                file_list = list(map(lambda x: x["key"], file_list))
        else:
            file_list = hlist_files([model_url])
        if file_list:
            target_file_path = ""
            for file_path in file_list:
                file_name = file_path.split(REMOTE_PATH_SEP)[-1]
                target_file_path = os.path.join(
                    EASYGUARD_MODEL_CACHE,
                    model_type,
                    sha256(pretrained_model_name_or_path),
                    file_name,
                )
                if not os.path.exists(target_file_path):
                    target_file_path = cache_file(
                        pretrained_model_name_or_path,
                        file_name,
                        model_url,
                        model_type,
                        if_cache=True,
                    )
            return REMOTE_PATH_SEP.join(target_file_path.split(REMOTE_PATH_SEP)[:-1])
        else:
            raise FileExistsError(f"not file found in server, please check the url `{model_url}`")
        # if isinstance(file_name, (str, set)):
        #     target_file_path = cache_file(
        #         pretrained_model_name_or_path, file_name, model_url, model_type
        #     )

        # elif isinstance(file_name, list):
        #     for file_name_ in file_name:
        #         target_file_path = cache_file(
        #             pretrained_model_name_or_path,
        #             file_name_,
        #             model_url,
        #             model_type,
        #         )
        # else:
        #     raise ValueError(
        #         f"the type of argument `file_name` must be one of [{str}, {set}, {list}]"
        #     )
        # return REMOTE_PATH_SEP.join(
        #     target_file_path.split(REMOTE_PATH_SEP)[:-1]
        # )
    ...


def list_pretrained_models() -> None:
    """list all pretrained models from archive.yaml"""
    model_archive = file_read(MODEL_ARCHIVE_PATH_BACKUP)
    model_archive_table = PrettyTable()
    filed_names = list()
    for key_, value_ in model_archive.items():
        filed_names += list(value_.keys())
    filed_names = list(sorted(set(filed_names)))
    model_archive_table.field_names = ["model_name"] + filed_names
    model_info_list = []
    for key_, value_ in model_archive.items():
        temp_ = [key_] + [value_.get(_, None) for _ in filed_names]
        model_info_list.append(temp_)
    # model_info_list = sorted(model_info_list, key=lambda x: x[0])
    model_archive_table.add_rows(model_info_list)
    # setting
    model_archive_table.sortby = "model_name"
    model_archive_table.align["model_name"] = "l"
    model_archive_table.align["description"] = "l"
    logger.info("\n" + str(model_archive_table))


def pretrained_model_archive_parse(
    model_name: str,
    model_config: Dict[str, Any],
    target_region: Optional[str] = "CN",
) -> Dict[str, Any]:
    """parse the ptrained model based the server config

    Parameters
    ----------
    model_name : str
        archive name, e.g., 'bert-base-uncased'
    model_config : Dict[str, Any]
        a dict from model archive
    target_region : Optional[str], optional
        CN, VA, CN/VA, by default "CN", deprecated

    Returns
    -------
    Dict[str, Any]
        a updated `model_config`
    """
    server_default = "tos"
    model_name_ = model_name.replace("-", "_")
    servers = model_config.get("server", None)

    if not servers:
        return model_config
    else:
        severs = servers.split("/")
        server = server_default if server_default in severs else severs[0]
        region = model_config.get("region", "CN")
        regions = region.split("/")
        target_region_ = target_region if target_region in regions else regions[0]
        region_index = REGION_MAPPING[target_region_]
        model_dir_remote = SERVER_MAPPING[server][region_index]
        if server == "tos":
            model_config["url_or_path"] = "tos"
        elif server == "hdfs":
            model_config["url_or_path"] = REMOTE_PATH_SEP.join([model_dir_remote, "models", model_name_])
        return model_config


def convert_model_weights(
    model_weights_path: str,
    prefix: str,
    output_name: Optional[str] = None,
    remove_old: Optional[bool] = False,
    map_location: Optional[str] = "cpu",
):
    """remove `prefix` of the weights' key in `model_weights_path`
    example:
    >>> path = '/root/.cache/easyguard/models/mdeberta_v2/4e7be1b4eb8d77a4a5932fa55ce000832b5deb76011d36820c5f78f5caf2ee83/pytorch_model.bin'
    >>> convert_model_weights(path, "backbone.")
    2023-02-01 20:05:29,211 INFO 58539 [/mnt/bn/ecom-govern-maxiangqian/dongjunwei/EasyGuard/easyguard/utils/auxiliary_utils.py:367]  start load weights from /root/.cache/easyguard/models/mdeberta_v2/4e7be1b4eb8d77a4a5932fa55ce000832b5deb76011d36820c5f78f5caf2ee83/pytorch_model.bin
    2023-02-01 20:05:30,704 INFO 58539 [/mnt/bn/ecom-govern-maxiangqian/dongjunwei/EasyGuard/easyguard/utils/auxiliary_utils.py:375]  save model into /root/.cache/easyguard/models/mdeberta_v2/4e7be1b4eb8d77a4a5932fa55ce000832b5deb76011d36820c5f78f5caf2ee83/pytorch_model_new.bin
    2023-02-01 20:05:33,728 INFO 58539 [/mnt/bn/ecom-govern-maxiangqian/dongjunwei/EasyGuard/easyguard/utils/auxiliary_utils.py:377]  convert successfully~

    Parameters
    ----------
    model_weights_path : str
        the path of the weights
    prefix : str
        the prefix, e.g., 'backbone.'
    output_name : Optional[str], optional
        the output name, e.g., '_v2', by default None
    remove_old : Optional[bool], optional
        if true, remove the old one, by default False
    map_location : Optional[str], optional
        used for `torch.load`, by default "cpu"

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    FileExistsError
        _description_
    """

    if not isinstance(prefix, str):
        raise ValueError(f"the argument `prefix` must be {str}")
    model_name = model_weights_path.split(os.path.sep)[-1]
    model_name_sep = model_name.split(".")
    if not output_name:
        model_name_sep[0] += "_new"
        output_name = ".".join(model_name_sep)
    elif output_name == model_name:
        raise ValueError(f"{output_name} is same as the orginal model")

    if remove_old:
        output_name = model_name
    output_path = model_weights_path[: -len(model_name)] + output_name
    if not os.path.exists(model_weights_path):
        raise FileExistsError(f"the file {model_weights_path} does not exist")
    logger.info(f"start load weights from {model_weights_path}")
    weights: OrderedDict = torch.load(model_weights_path, map_location=map_location)
    for key_ in list(weights.keys()):
        if key_.startswith(prefix):
            key_new = key_.repalce(prefix, "", 1)
            weights[key_new] = weights.pop(key_)
    logger.info(f"save model into {output_path}")
    torch.save(weights, output_path)
    logger.info(f"convert successfully~")


# all code from titan
def get_configs(**kwargs):
    pretrained_config = {}
    pretrained_config["tos_helper"] = kwargs.pop("tos_helper", None)
    pretrained_config["pretrained_version"] = kwargs.pop("pretrained_version", None)
    pretrained_config["pretrained_uri"] = kwargs.pop("pretrained_uri", None)
    pretrained_config["local_rank"] = kwargs.pop("local_rank", None)
    pretrained_config["rank"] = kwargs.pop("rank", None)
    pretrained_config["rank_zero_only"] = kwargs.pop("rank_zero_only", True)
    pretrained_config["backend"] = kwargs.pop("backend", "titan")

    model_config = kwargs.copy()
    if kwargs.get("features_only", None) is None:
        model_config["features_only"] = False

    return pretrained_config, model_config


# main code from titan
def load_pretrained_model_weights(
    model,
    model_path: str,
    strict: Optional[bool] = False,
    rm_prefix: Optional[str] = None,
    change_prefix: Optional[List[tuple]] = None,
    **kwargs,
):
    """load pretrained model weights

    Parameters
    ----------
    model : _type_
        the model
    model_path : str
        the path of model weights
    strict : Optional[bool], optional
        the arguments about torch, by default False
    rm_prefix : Optional[str], optional
        if not none, will remove the prefixs of the model weighs and then load model weighs, by default None
    """
    if model_path is None:
        raise FileExistsError("no file for the model weight")

    map_location = kwargs.pop("map_location", "cpu")

    # load weights
    if not model_path.startswith("hdfs://"):
        ckpt = torch.load(model_path, map_location=map_location)
    else:
        with hdfs_open(model_path, "rb") as hdfs_reader:
            logger.info(f"start to read `{model_path}` file from hdfs")
            state_dict = io.BytesIO(hdfs_reader.read())
            logger.info(f"read `{model_path}` successfully")
            ckpt = torch.load(state_dict, map_location=map_location)

    if ckpt.get("state_dict", None):
        state_dict = ckpt["state_dict"]
    elif ckpt.get("state_dict_ema", None):
        state_dict = ckpt["state_dict_ema"]
    elif ckpt.get("model", None):
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    if rm_prefix:
        for key_ in list(state_dict.keys()):
            if key_.startswith(rm_prefix):
                key_new = key_.replace(rm_prefix, "", 1)
                state_dict[key_new] = state_dict.pop(key_)
    # if change_prefix:

    with HiddenPrints():
        keys = model.load_state_dict(state_dict, strict=strict)

    @typecheck(list, list, dict)
    def table_formatter(
        field_names: List[str],
        values: List[List[str]],
        align: Dict[str, str],
        sortby: Optional[int] = 0,
    ):
        table = PrettyTable()
        table.field_names = field_names
        table.add_rows(values)
        for key_ in field_names:
            if key_ in align:
                table.align[key_] = align[key_]
        table.sortby = field_names[sortby]
        return str(table)

    if len(keys.missing_keys) > 0:
        field_names = ["missing_keys"]
        values = list(map(lambda x: [x], keys.missing_keys))
        align = {"missing_keys": "l"}

        logger.warning(f"=> Pretrained: missing_keys: \n{table_formatter(field_names, values, align)}\n")
    else:
        logger.info(f"Pretrained: no missing_keys.")
    if len(keys.unexpected_keys) > 0:
        field_names = ["unexpected_keys"]
        values = list(map(lambda x: [x], keys.unexpected_keys))
        align = {"unexpected_keys": "l"}
        logger.warning(f"=> Pretrained: unexpected_keys: \n{table_formatter(field_names, values, align)}\n")
    else:
        logger.info(f"Pretrained: no unexpected_keys.")

    logger.info(f"=> Pretrained: load pretrained model with strict={strict}")
