# import os

from easyguard.utils.tos_helper import TOSHelper

# # 模型上传到TOS
# tos_help = TOSHelper(
#     bucket="ecom-govern-easyguard-zh",
#     access_key="SHZ0CK8T8963R1AVC3WT",
#     service="toutiao.tos.tosapi"
# )

# file_path = "./chinese-alpaca-2-13b-1008/"
# for filename in os.listdir(file_path):
#     print(filename)
#     input_path = os.path.join(file_path, filename)
#     tos_help.upload_model_to_tos(
#         input_path,
#         filename=filename,
#         directory="chinese-alpaca-2-13b-1008",
#         force_overwrite=True,
#     )


# 从TOS下载模型到本地
tos_help = TOSHelper(
    bucket="ecom-govern-easyguard-zh",
    access_key="SHZ0CK8T8963R1AVC3WT",
    service="toutiao.tos.tosapi",
)
for item in tos_help.list_files(directory="chinese-alpaca-2-13b-1008"):
    print(item)
    tos_help.download_model_from_tos(
        filename=item,
        output_path=item,
    )
