# Very heavily inspired by:
# https://code.byted.org/lab/titan/blob/master/titan/utils/tos_helper.py
# more information about TOS SDK, please refer to:
# https://bytedance.feishu.cn/wiki/wikcnclPFegm60un88KX7VIiR7b#

import json
import os
import sys

import bytedtos
from cruise.utilities.logger import get_cruise_logger

logger = get_cruise_logger()

TOS_IDC = ("lf", "lq", "hl", "alisg", "sgcomm1", "maliva")

# 100M is a good choice for TOS
CHUNK_SIZE = 1 * 1024 * 1024 * 100


def _is_large_file(buf):
    return len(buf) > CHUNK_SIZE


class TOSHelper(object):
    """Provide convenient methods for interactions with TOS.
    `create_model` API needs tos helper instance to perform operations to/from TOS buckets.

     Args:
         bucket (str):
             name for tos buckets.
             Public buckets: `titan-modelzoo-public`, `titan-modelzoo-public-sg`, `titan-modelzoo-public-us`
             For private buckets please refer to https://bytedance.sg.feishu.cn/docx/doxcnoHPLokn89oxDYWDva3hJ6g.
         access_key (str): the TOS access key.
         cluster (str, optional): The cluster TOS bucket is on. Defaults to 'default'.
         timeout (int, optional): Request timeout in second. Defaults to 200.
         idc (str, optional):
             The data-center. Defaults to ''.
             Available idcs are 'lf', 'lq', 'hl', 'alisg', 'sgcomm1', 'maliva'.
             Note that you usually only need default option, however, for sg region, please
             use `sgcomm1`.
         connect_timeout (int, optional): Connection timeout in second. Defaults to 30.

     Raises:
         ValueError: If idc is not valid.
    """

    def __init__(self, bucket, access_key, cluster="default", timeout=2000, idc="", connect_timeout=300, **kwargs):
        if idc and idc not in TOS_IDC:
            raise ValueError(f"Expected idc in {TOS_IDC}, got {idc}")
        self.client = bytedtos.Client(
            bucket, access_key, cluster=cluster, timeout=timeout, idc=idc, connect_timeout=connect_timeout, **kwargs
        )

    @property
    def bucket(self):
        return self.client.bucket

    @property
    def cluster(self):
        return self.client.cluster

    @property
    def idc(self):
        return self.client.idc

    @property
    def timeout(self):
        return self.client.timeout

    def _join_directory(self, filename, directory=""):
        if directory:
            filename = "%s/%s" % (directory, filename)
        return filename

    def exists(self, filename, directory="", verbose=False):
        filename = self._join_directory(filename, directory=directory)
        try:
            rsp = self.client.head_object(filename)
            if verbose:
                logger.info(f"File {filename} EXISTS in TOS bucket {self.bucket}")
            return rsp
        except Exception:
            # file not exists
            if verbose:
                logger.info(f"File {filename} NOT EXISTS in TOS bucket {self.bucket}")
            return None

    def download(self, filename, directory="", verbose=False):
        filename = self._join_directory(filename, directory=directory)
        try:
            if not self.exists(filename):
                raise ValueError(f"File {filename} not found in TOS bucket {self.bucket}")
            rsp = self.client.get_object(filename)
            data = rsp.data
            if verbose:
                logger.info(f"Downloaded file {filename} from TOS bucket " f"{self.bucket}")
            return data
        except Exception as e:
            msg = f"Failed to download file {filename} " f"from TOS bucket {self.bucket}: {e}"
            logger.error(msg)
            raise ValueError(msg)

    def download_model_from_tos(self, filename, output_path, directory="", verbose=False):
        filename = self._join_directory(filename, directory=directory)
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        try:
            data = self.download(filename)
            with open(output_path, "wb") as writer:
                writer.write(data)
        except:  # noqa: E722
            logger.info(f"Model path {output_path} already exists, " f"skip downloading.")
            return

        if verbose:
            logger.info(f"Downloaded file {filename} from TOS bucket " f"{self.bucket} to {output_path}")

    def _upload(self, filename, buf, verbose=False):
        if not _is_large_file(buf):
            if verbose:
                logger.info("upload file directly to TOS")
            self.client.put_object(filename, buf)
            return
        if verbose:
            logger.info("upload large file to TOS")
        total_size = len(buf)
        chunk_count = total_size / CHUNK_SIZE
        start = 0
        remaining_size = total_size
        cached_list = []
        cached_size = 0
        rsp = self.client.init_upload(filename)
        data = json.loads(rsp.data.decode("utf-8"))
        upload_id = data["payload"]["uploadID"]
        current_part = 1
        parts = []

        while chunk_count >= 0:
            if verbose:
                logger.info(f"upload file from Byte{start} to " f"Byte{start + CHUNK_SIZE}")
            data = buf[start : start + CHUNK_SIZE]
            cached_list.append(data)
            cached_size += len(data)
            remaining_size -= len(data)
            if cached_size > CHUNK_SIZE and remaining_size > CHUNK_SIZE:
                data = b"".join(cached_list)
                part_rsp = self.client.upload_part(filename, upload_id, str(current_part), data)
                part = part_rsp.part_number
                parts.append(part)
                current_part += 1
                cached_list[:] = []
                cached_size = 0
            start += CHUNK_SIZE
            chunk_count -= 1
        data = b"".join(cached_list)
        part_rsp = self.client.upload_part(filename, upload_id, str(current_part), data)
        part = part_rsp.part_number
        parts.append(part)
        current_part += 1
        cached_list[:] = []
        cached_size = 0
        self.client.complete_upload(filename, upload_id, parts)

    def upload(self, buf, filename, directory="", force_overwrite=False, verbose=False):
        filename = self._join_directory(filename, directory=directory)
        try:
            if self.exists(filename):
                if force_overwrite:
                    logger.warning(
                        f"File {filename} already exists in TOS bucket "
                        f"{self.bucket} and force_overwrite=True, try to "
                        f"overwrite this file"
                    )
                elif verbose:
                    logger.warning(
                        f"File {filename} already exists in TOS bucket "
                        f"{self.bucket} and force_overwrite=False, "
                        f"cancel upload"
                    )
                    return False

            self._upload(filename, buf, verbose)
            if not self.exists(filename):
                raise ValueError("file existence validation failed, maybe failed to upload")
            if verbose:
                logger.info(f"Uploaded data to TOS bucket:{self.bucket} " f"with filename:{filename}")
            return True
        except Exception as e:
            msg = f"Failed to upload data to TOS bucket {self.bucket} " f"with filename {filename}: {e}"
            logger.error(msg)
            raise ValueError(msg)

    def upload_model_to_tos(self, input_path, filename, directory="", force_overwrite=False, verbose=False):
        if not os.path.isfile(input_path):
            raise IOError("Input path {} not found".format(input_path))
        filename = self._join_directory(filename, directory=directory)
        if self.exists(filename) and not force_overwrite:
            logger.warning(
                f"File {filename} already exists in TOS bucket "
                f"{self.bucket} and force_overwrite=False, cancel upload"
            )
            return False
        else:
            buf = open(input_path, "rb").read()
            success = self.upload(buf, filename, force_overwrite=force_overwrite, verbose=verbose)
            if success and verbose:
                logger.info(f"Uploaded file {input_path} to TOS bucket " f"{self.bucket} with filename {filename}")
        return success

    def delete(self, filename, directory="", verbose=False):
        filename = self._join_directory(filename, directory=directory)
        try:
            if not self.exists(filename):
                if verbose:
                    logger.warning(f"File {filename} does not exist in TOS bucket " f"{self.bucket}, cancel delete")
            else:
                self.client.delete_object(filename)
                if self.exists(filename):
                    raise ValueError("file still exists after deleting")
                if verbose:
                    logger.info(f"Deleted file {filename} from TOS bucket " f"{self.bucket}")
        except Exception as e:
            msg = "Failed to delete file {} in TOS bucket {}: {}".format(filename, self.bucket, e)
            logger.error(msg)
            raise ValueError(msg)

    def list_files(self, file_prefix="", directory="", delimiter="/"):
        """list all files with the name prefix as file_prefix in directory"""
        if not file_prefix and not directory:
            raise ValueError("None of input file_prefix and directory is provided")
        if file_prefix:
            file_prefix = os.path.join(directory, file_prefix) if directory else file_prefix
            logger.info("List TOS files with filename prefix {}".format(file_prefix))
        else:
            if not directory.endswith(delimiter):
                directory += delimiter
            logger.info('List TOS files in "directory" {}'.format(directory))
            file_prefix = directory
        # send requests
        start_after = ""
        max_num_keys = 1000
        has_more = True
        subdirs = []
        curr_file_prefix = file_prefix
        while has_more or subdirs:
            try:
                rsp = self.client.list_prefix(curr_file_prefix, delimiter, start_after, max_num_keys)
                if sys.version_info.major >= 3 and isinstance(rsp, bytes):
                    # json only loads string object, not bytes object,
                    # need a conversion here
                    rsp = rsp.decode("utf-8")
                rsp = rsp.json
                code = rsp.get("success", 1)
                if code != 0:
                    raise ValueError(f"Return code {code} != 0")
                # check if there are subdirs starting with the same
                # curr_file_prefix
                if rsp["payload"]["commonPrefix"]:
                    subdirs += rsp["payload"]["commonPrefix"]
                has_more = rsp["payload"]["isTruncated"]
                start_after = rsp["payload"]["startAfter"]
                if rsp["payload"]["objects"]:
                    for obj in rsp["payload"]["objects"]:
                        logger.info(f'Filename: {obj["key"].split(delimiter)[-1]}')
                        yield obj["key"]
                if not has_more and subdirs:
                    # no more file for current level of directory-tree,
                    # move to next dir in subdirs
                    curr_file_prefix = subdirs.pop()
                    has_more = True
            except Exception as e:
                raise ValueError(
                    f"Failed to list files with filename prefix / directory"
                    f" {file_prefix} and delimiter {delimiter}: {e}"
                )

    def list_dir(self, directory, delimiter="/"):
        """list all files in directory"""
        return self.list_files(directory=directory, delimiter=delimiter)

    def list_subfolders(self, directory, delimiter="/"):
        """list all subfolders in a directory, only traverse at one level"""
        start_after = ""
        max_num_keys = 1000
        has_more = True
        if not directory.endswith(delimiter):
            directory += delimiter
        logger.info(f'List "subfolders" in TOS "directory" {directory}')
        curr_file_prefix = directory
        while has_more:
            try:
                rsp = self.client.list_prefix(curr_file_prefix, delimiter, start_after, max_num_keys)
                if sys.version_info.major >= 3 and isinstance(rsp, bytes):
                    # json only loads string object, not bytes object,
                    # need a conversion here
                    rsp = rsp.decode("utf-8")
                rsp = rsp.json
                code = rsp.get("success", 1)
                if code != 0:
                    raise ValueError(f"Return code {code} != 0")
                has_more = rsp["payload"]["isTruncated"]
                start_after = rsp["payload"]["startAfter"]
                if rsp["payload"]["commonPrefix"]:
                    for obj in rsp["payload"]["commonPrefix"]:
                        if obj.endswith(delimiter):
                            obj = obj[:-1]
                        yield obj.split(delimiter)[-1]
                        logger.info(f"Subfolder: {obj.split(delimiter)[-1]}")
            except Exception as e:
                raise ValueError(
                    f"Failed to list files with filename directory {directory}" f" and delimiter {delimiter}: {e}"
                )
