import importlib
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional

from .. import EASYGUARD_MODEL_CACHE
from ..utils import HDFS_HUB_CN, HDFS_HUB_VA, REGION_MAPPING, REMOTE_PATH_SEP, hmget, logging, sha256

logger = logging.get_logger(__name__)

HUB_MAPPING = OrderedDict([["hdfs", "HdfsHub"], ["byteNAS", "ByteNASHub"]])


class HubBase(ABC):
    """all kinds of hub

    Parameters
    ----------
    ABC : _type_
        _description_
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_file(self):
        ...

    @abstractmethod
    def download_file(self):
        ...


# TODO: 实现本地数据中心
class AutoHubClass:
    kwargs = None
    hub_class = None

    def __init__(self) -> None:
        """a factory, just used for instantiate a hub class and take over all
           properties and methods of the template object

        Parameters
        ----------
        server_name : str
            server type in `HUB_MAPPING`
        archive_name : str
            archive_name in `archive.yaml`
        model_type : str
            model name in `models.yaml`

        Returns
        -------
        _type_
            _description_
        """
        server_name: str = self.kwargs.pop("server_name", None)
        archive_name: str = self.kwargs.pop("archive_name", None)
        model_type: str = self.kwargs.pop("model_type", None)
        module_name = HUB_MAPPING.get(server_name, None)
        if not module_name:
            return ValueError(f"not support {server_name}")
        module = importlib.import_module(__name__)
        class_module = getattr(module, module_name, None)
        if not class_module:
            class_instance = class_module(
                archive_name=archive_name,
                model_type=model_type,
                **self.kwargs,
            )
            for key_ in dir(class_instance):
                if not key_.startswith("__"):
                    setattr(self, key_, getattr(class_instance, key_))


# TODO (junwei.Dong): hub中心如何更加有效的给模型提供下载文件的功能，基于__mro__继承链的多继承很难实现，多继承最好的方式还是用来做功能组合
class HdfsHub(HubBase):
    hub_urls = [HDFS_HUB_CN, HDFS_HUB_VA]

    def __init__(self, archive_name: str, model_type: str, region: str, *args, **kwargs) -> None:
        """get file from the hdfs hub

        Parameters
        ----------
        archive_name : str
            archive_name in `archive.yaml`
        model_type : str
            model name in `models.yaml`
        region : str
            available region
        """
        self.archive_name = archive_name
        self.archive_name_ = archive_name.replace("-", "_")
        self.model_type = model_type
        self.hub_url = self.hub_urls[REGION_MAPPING[region]]
        self._model_dir_remote = REMOTE_PATH_SEP.join([self.hub_url, "models", self.archive_name_])
        self._model_dir_local = os.path.join(EASYGUARD_MODEL_CACHE, model_type, sha256(archive_name))

    def get_file(self, file_name: str, force: Optional[bool] = False):
        """get file from remote hdfs model directory

        Parameters
        ----------
        file_name : str
            file which exist in hdfs model directory
        force : Optional[bool], optional
            if true, remove the local same file, by default False

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        FileExistsError
            _description_
        """
        file_path_local = os.path.join(self._model_dir_local, file_name)
        file_path_remote = REMOTE_PATH_SEP.join([self._model_dir_remote, file_name])
        if self.download_file(file_path_remote, file_path_local, force):
            return file_path_local
        raise FileExistsError(f"the {file_name} does not exist locally and remotely, please check~")

    @classmethod
    def download_file(cls, hdfs_url: str, target_path: str, force: Optional[bool] = False) -> bool:
        """download file

        Parameters
        ----------
        hdfs_url : str
            hdfs url of file
        target_path : str
            local directory
        force : Optional[bool], optional
            if true, remove the local same file, by default False

        Returns
        -------
        bool
            _description_
        """
        if os.path.exists(target_path) and force:
            os.remove(target_path)
        target_dir_path = os.path.dirname(target_path)
        logger.info(f"start to download file {hdfs_url}")
        hmget([hdfs_url], target_dir_path)
        if os.path.getsize(target_path) == 0:
            os.remove(target_path)
            logger.warning(f"fail to download `{hdfs_url}`, please check the network or the remote file path")
            return False
        else:
            logger.info("download successfully~")
            return True


class ByteNASHub(HubBase):
    def __init__(self) -> None:
        super().__init__()

    def get_file(self):
        ...

    def download_file(self):
        ...
