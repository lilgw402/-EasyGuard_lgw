import os

import bytedenv
import pytest

from easyguard.utils.tos_helper import TOSHelper

idc_name = bytedenv.get_idc_name()


def test_tos_helper():
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
    # 测试 list_files 函数的功能
    with pytest.raises(ValueError):
        for file_name in tos_helper.list_files():
            print(file_name)
    for file_name in tos_helper.list_files(directory="deberta_base_6l"):
        print(file_name)
    for file_name in tos_helper.list_files(file_prefix="deberta_base_6l"):
        print(file_name)
    print("=" * 20 + "test list_files finished" + "=" * 20)

    # 测试 list_dir 函数的功能
    for file_name in tos_helper.list_dir(directory="deberta_base_6l"):
        print(file_name)
    print("=" * 20 + "test list_dir finished" + "=" * 20)

    # 测试 list_subfolders 函数的功能
    for file_name in tos_helper.list_subfolders(directory="deberta_base_6l"):
        print(file_name)
    print("=" * 20 + "test list_subfolders finished" + "=" * 20)

    # 测试 upload_model_to_tos 函数的功能
    input_path = "test.txt"
    with open(input_path, "w") as f:
        f.write("Hello, world!\n")
    tos_helper.upload_model_to_tos(input_path, filename=input_path, force_overwrite=True, verbose=True)
    print("=" * 20 + "test upload_model_to_tos finished" + "=" * 20)

    # 测试 exists 函数的功能
    rsp = tos_helper.exists(input_path)
    assert rsp is not None
    print("=" * 20 + "test exists finished" + "=" * 20)

    # 测试 download_model_from_tos 函数的功能
    tos_helper.download_model_from_tos(filename="config.yaml", output_path="config.yaml", directory="deberta_base_6l")
    assert os.path.exists("config.yaml")
    print("=" * 20 + "test download_model_from_tos finished" + "=" * 20)

    # 测试 delete 函数的功能
    tos_helper.delete(filename=input_path)
    print("=" * 20 + "test delete finished" + "=" * 20)

    os.remove(input_path)
    os.remove("config.yaml")


if __name__ == "__main__":
    pytest.main(["-s", "-v"])
