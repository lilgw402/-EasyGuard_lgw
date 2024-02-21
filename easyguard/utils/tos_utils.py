import json
import os
from typing import Dict, Optional, Union

import bytedenv
import bytedtos
import requests
from prettytable import PrettyTable
from progressbar import *

from easyguard.modelzoo.config import MODEL_ARCHIVE_CONFIG, MODEL_ARCHIVE_PATH_BACKUP, TOS_FILES_PATH

from . import AK_CN, AK_VA, BUCKET_CN, BUCKET_VA, CDN_VA, ENDPOINT_CN, ENDPOINT_VA, TOS_HTTP_CN, TOS_HTTP_VA
from .logging import get_logger
from .yaml_utils import json2yaml, load_yaml

logger = get_logger(__name__)

TIMEOUT = 500
TIMEOUT_CONNECT = 500

IDC_MAPPING = {"hl": "va", "maliva": "va"}


class TOS:
    kb: int = 1024
    mb: int = 1024**2
    gb: int = 1024**3
    size_threshold: int = 20  # when size < 20MB, upload directly

    def __init__(
        self,
        bucket_name: str = BUCKET_CN,
        access_key: str = AK_CN,
        endpoint: str = ENDPOINT_CN,
        timeout: int = TIMEOUT,
        timeout_connect: int = TIMEOUT_CONNECT,
        http_cdn_va: str = TOS_HTTP_VA,
        http_cn: str = TOS_HTTP_CN,
        just_download: bool = True,
        just_own: bool = False,
    ) -> None:
        """create a tos client and set a cdn url for va

        Parameters
        ----------
        bucket_name : str, optional
            the target tos bucket in china, by default BUCKET_CN
        access_key : str, optional
            the ak for target bucket, which can be found in tos website, by default AK_CN
        endpoint : str, optional
            same as `access_key`, by default ENDPOINT_CN
        timeout : int, optional
            how long total time, by default TIMEOUT
        timeout_connect : int, optional
            how long onect time, by default TIMEOUT_CONNECT
        http_cdn_va : str, optional
            cdn url for request in va region, by default TOS_HTTP_VA
        just_download: bool, optional
            if just download, not connect to cn tos, by default True
        """
        if not just_own:
            if "va" not in self.get_idc():
                self.bucket_name = BUCKET_CN
                self.endpoint_ = ENDPOINT_CN
            else:
                self.bucket_name = BUCKET_VA
                access_key = AK_VA
                self.endpoint_ = ENDPOINT_VA
        self.timeout_ = timeout
        self.http_va_ = http_cdn_va
        self.http_cn_ = http_cn
        self.tos_client = None
        self.tos_client = bytedtos.Client(
            self.bucket_name,
            access_key,
            endpoint=self.endpoint_,
            timeout=timeout,
            connect_timeout=timeout_connect,
        )

    @property
    def bucket(self):
        return self.bucket_name

    @property
    def timeout(self):
        return self.timeout_

    @property
    def http_cn(self):
        return self.http_cn_

    @property
    def http_va(self):
        return self.http_va_

    @property
    def endpoint(self):
        return self.endpoint_

    @classmethod
    def get_idc(cls) -> str:
        """get the region

        Returns
        -------
        str
            region, cn[hl, lq...] or va
        """
        return bytedenv.get_idc_name()

    @classmethod
    def request_file(
        cls,
        url: str,
        target_path: str,
        part_number: int = 100,
        timeout_time: int = 20,
    ):
        """support breakpoint transfer from a website such as tos

        Parameters
        ----------
        url : str
            a url which can be got
        target_path : str
            a local path to save
        part_number : int, optional
            multi parts, by default 100
        timeout_time : int, optional
            if break, the max number of connections, by default 20

        """
        widgets = [
            "Progress: ",
            Percentage(),
            " ",
            Bar("#"),
            " ",
            Timer(),
            " ",
            ETA(),
            " ",
            FileTransferSpeed(),
        ]

        total_length = int(requests.head(url).headers.get("content-length", -1))
        temp_size = 0
        default_size = 100
        pbar = None
        timeout_time_ = 0
        if total_length == -1:
            pbar = ProgressBar(widgets=widgets, maxval=default_size).start()
            with requests.get(url) as r:
                with open(target_path, "wb") as f:
                    f.write(r.content)
                    pbar.update(default_size)
        else:
            while True:
                if os.path.exists(target_path):
                    temp_size = os.path.getsize(target_path)
                    if temp_size >= total_length:
                        if pbar:
                            pbar.update(total_length)
                            pbar.finish()
                        else:
                            logger.warning(f"target file {target_path} exist, please check~")
                        break
                if timeout_time_ >= timeout_time:
                    if temp_size < total_length:
                        logger.warning(f"\n the size of `{url}` is not same as the downloaded one~")
                    break

                headers = {
                    "Range": f"bytes={temp_size}-",
                    "User-Agent": "bytedance-ecom-govern",
                }
                if total_length < 1000:
                    part_size = total_length
                else:
                    part_size = int(total_length / part_number)

                with requests.get(url, stream=True, headers=headers) as r:
                    # r.raise_for_status()
                    if not pbar:
                        pbar = ProgressBar(widgets=widgets, maxval=total_length).start()
                    receive_size = temp_size
                    try:
                        with open(target_path, "ab") as f:
                            for chunk in r.iter_content(part_size):
                                receive_size += part_size
                                pbar.update(receive_size if receive_size < total_length else total_length)
                                if chunk:
                                    f.write(chunk)
                    except:
                        ...
                timeout_time_ += 1

    def update_local(self):
        archive_dict = load_yaml(MODEL_ARCHIVE_PATH_BACKUP)
        model_dir_dict = dict()
        for key_ in archive_dict.keys():
            key_ = key_.replace("-", "_")
            try:
                files_ = list(
                    map(
                        lambda x: {
                            "key": x["key"],
                            "lastModified": x["lastModified"],
                            "size": x["size"],
                        },
                        self.ls(key_, if_log=False),
                    )
                )
                model_dir_dict[key_] = files_
            except:
                ...

        json2yaml(TOS_FILES_PATH, model_dir_dict)

    def size_show(self, size: int):
        div_, unit_ = 1, "B"
        if size / self.gb > 1:
            div_, unit_ = self.gb, "GB"
        elif size / self.mb > 1:
            div_, unit_ = self.mb, "MB"
        elif size / self.kb > 1:
            div_, unit_ = self.kb, "KB"

        return (round(size / div_, 2), unit_) if div_ != 1 else (size, unit_)

    def _get_obj_info(self, key_name: str):
        file_list = []
        if "." in key_name:
            file_list.append(key_name)
        else:
            file_objs = self.ls(key_name, if_log=False)
            for file_ in file_objs:
                file_list.append(file_["key"])
        return file_list

    def _put_file_small(self, file_path: str, name: str):
        with open(file_path, "rb") as f:
            file_data = f.read()
        try:
            self.tos_client.put_object(name, file_data)
            return True
        except bytedtos.TosException as e:
            logger.info(e)
            return False

    def _put_file_large(self, file_path: str, name: str):
        # todo: can use multipart upload, should use io to deal with the byte data, ref: https://docs.python.org/zh-cn/3.9/library/io.html
        ...

    def exist(self, name: str):
        """if file or directory exist, return True, else, return False

        Parameters
        ----------
        name : str
            file name or directory name

        Returns
        -------
        bool
            True: exist, False: None
        """
        if "." in name:
            if self.tos_client:
                try:
                    self.tos_client.head_object(name)
                    return True
                except:
                    ...
            else:
                if requests.head("/".join([self.http_va_, name])).status_code == 200:
                    return True
                else:
                    return False
        else:
            if self.tos_client:
                try:
                    dir_obj = self.ls(name, if_log=False)
                    if dir_obj:
                        return True
                except:
                    ...
            else:
                return name in self.file_info_local

        return False
        ...

    def ls(self, dir_name: str, max_number: int = 10, if_log: bool = True):
        """list files in directory in tos

        Parameters
        ----------
        dir_name : str
            a dirname in tos
        max_number : int, optional
            the max number of file, by default 10
        if_log: bool
            if true, print table, by default True
        """

        def object_parse(file_obj: Dict[str, str]):
            name = file_obj["key"].split("/")[-1]
            modified_time = file_obj["lastModified"]
            size, unit_name = self.size_show(file_obj["size"])
            return [name, f"{size}{unit_name}", modified_time]

        if self.tos_client:
            response = self.tos_client.list_prefix(f"{dir_name}/", "", "", max_number).data
            dir_obj = json.loads(response)
            object_arr = dir_obj["payload"]["objects"]
        else:
            object_arr = self.file_info_local[dir_name]

        if not object_arr:
            return None
        object_arr_new = list(map(object_parse, object_arr))
        if if_log:
            object_table = PrettyTable()
            object_table.field_names = ["name", "size", "modified time"]
            object_table.add_rows(object_arr_new)
            (
                object_table.align["name"],
                object_table.align["size"],
                object_table.align["modified time"],
            ) = ("l", "l", "l")
            logger.info("\n" + str(object_table))
        return object_arr

    def get(self, key_name: str, save_dir: str = "./", verbose: bool = True):
        assert self.exist(key_name), f"please check the name exist in tos[{self.bucket}]"
        get_files = self._get_obj_info(key_name)
        local_files = list(map(lambda x: os.path.join(save_dir, x.split("/")[-1]), get_files))
        save_dir_ = os.path.dirname(local_files[0])
        os.makedirs(save_dir_, exist_ok=True)
        for index, file_ in enumerate(get_files):
            if verbose:
                logger.info(f"start to download file `{file_}`...")

            respone = self.tos_client.get_object(file_)
            with open(local_files[index], "wb") as writer:
                writer.write(respone.data)

        if verbose:
            logger.info(f"all the files are saved in {save_dir_}")

    def get_cdn(self, key_name: str, save_dir: str = "./", verbose: bool = True):
        """can get file or directory from tos[cn/sg], just support txt, json, yaml, torch model

        Parameters
        ----------
        key_name : str
            file or directory
        save_dir : str
           directory
        verbose: bool
            log or not

        Example:
        >>> tos = TOS()
        >>> tos.get("fashionxlm_moe", './')
        >>> tos.get("fashionxlm_moe/config.json", './fashionxlm_moe')
        """

        # def decode(data: bytes, file_name: str, file_name_local: str):
        #     """decode data into different type

        #     Parameters
        #     ----------
        #     file_name : str
        #         a name of file
        #     data : bytes
        #         tos data
        #     file_name_local: str
        #         local file path

        #     Returns
        #     -------
        #     _type_
        #         _description_
        #     """
        #     if not data:
        #         return

        #     if file_name.endswith(".json"):
        #         json_decode(data, file_name_local)
        #     elif file_name.endswith(".yaml"):
        #         yaml_decode(data, file_name_local)
        #     elif file_name.endswith(".txt"):
        #         txt_decode(data, file_name_local)
        #     else:
        #         common_decode(data, file_name_local)

        # def json_decode(data: bytes, file_name_local: str):
        #     with open(file_name_local, "w") as f:
        #         json.dump(json.loads(data), f)

        # def txt_decode(data: bytes, file_name_local: str):
        #     with open(file_name_local, "w") as f:
        #         f.write(data.decode("utf-8"))

        # def yaml_decode(data: bytes, file_name_local: str):
        #     with open(file_name_local, "w") as f:
        #         yaml.dump(yaml.load(data, yaml.FullLoader), f)

        # def common_decode(data: bytes, file_name_local: str):
        #     with open(file_name_local, "wb") as f:
        #         f.write(data)

        assert self.exist(key_name), f"please check the name exist in tos[{self.bucket}]"

        idc = self.get_idc()

        region = IDC_MAPPING.get(idc, "va")

        get_files = self._get_obj_info(key_name)
        local_files = list(map(lambda x: os.path.join(save_dir, x.split("/")[-1]), get_files))
        save_dir_ = os.path.dirname(local_files[0])
        os.makedirs(save_dir_, exist_ok=True)
        for index, file_ in enumerate(get_files):
            # data = None
            if verbose:
                logger.info(f"start to download file `{file_}`:\n")

            tos_http = TOS_HTTP_CN
            if region == "cn":
                # data = self.tos_client.get_object(file_).data
                tos_http = TOS_HTTP_CN
            else:
                tos_http = TOS_HTTP_VA

            url = "/".join([tos_http, file_])
            self.request_file(url, local_files[index])

            # if verbose:
            #     logger.info(f"download file `{file_}` successfully!")
            # decode(data, file_, local_files[index])
        if verbose:
            logger.info(f"all the files are saved in {save_dir_}")

    def put(self, path: str, dir_name: Union[None, str] = None):
        """put a file or directory to a tos

        Parameters
        ----------
        path : str
            file path or directory path
        dir_name: Union[None, str]
            available when `path` is a directory, tos directory, if set, the tos dirname will change, otherwise, the dir name will be set

        Example:
        >>> tos = TOS()

        >>> tos.put("./fashionxlm_moe")
        TOS will save fashionxlm_moe directory

        >>> tos.put("./fashionxlm_moe", "fashionxlm_moe_v1")
        TOS will save fashionxlm_moe directory with the name fashionxlm_moe_v1

        >>> tos.put("./test.txt")
        """
        assert os.path.exists(path), f"{path} does not exist, please check!"
        file_prefixs = path.split(os.path.sep)
        file_name = file_prefixs[-1]
        upload_files = []
        if not os.path.isdir(path):
            upload_files.append((path, file_name))
        else:
            file_name = file_name if not dir_name else dir_name
            file_list = os.listdir(path)
            file_name_list = list(map(lambda x: f"{file_name}/{x}", file_list))
            file_path_list = list(map(lambda x: f"{path}/{x}", file_list))
            upload_files += list(zip(file_path_list, file_name_list))

        error_files = []
        for file_ in upload_files:
            file_path, file_name = file_
            file_size, unit_name = self.size_show(os.path.getsize(file_path))
            logger.info(f"file `{file_[0]}` [{file_size}{unit_name}] start to upload...")
            file_size_mb = os.path.getsize(file_path) / self.mb
            if file_size_mb < self.size_threshold:
                statue = self._put_file_small(file_path, file_name)
            else:
                statue = self._put_file_small(file_path, file_name)
            if not statue:
                error_files.append(file_[0])
            else:
                logger.info(f"file `{file_[0]}` upload successfully!")
        if len(error_files) == 0:
            logger.info(f"all files upload successfully!")
        else:
            logger.info(f"failed to upload these files: {error_files}, they may do not exist, please check!")
        self.update_local()
        return True

    def rm(self, name: str):
        """delete a file or a directory

        Parameters
        ----------
        name : str
            file name or directory name

        Example:
        >>> tos = TOS()
        >>> tos.rm("./fashionxlm_moe")
        >>> tos.rm("./fashionxlm_moe/config.json")
        """
        assert self.exist(name), f"please check the name exist in tos[{self.bucket}]"

        rm_files = []
        if "." in name:
            rm_files.append(name)
        else:
            file_objs = self.ls(name, if_log=False)
            for file_ in file_objs:
                rm_files.append(file_["key"])

        unrm_files = []
        for file_ in rm_files:
            unrm_files.append(file_)
            try:
                response = self.tos_client.delete_object(file_)
                if response.status_code == 204:
                    logger.info(f"delete {file_} successfully~")
                    unrm_files.pop()
            except:
                ...
        self.update_local()
        if len(unrm_files) == 0:
            logger.info(f"delete all files successfully!")
            return True
        else:
            logger.info(f"failed to delete these files: {unrm_files}, they may do not exist, please check tos!")
            return False

        ...
