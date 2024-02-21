# import os

from easyguard.utils.tos_helper import TOSHelper

# 模型上传到TOS
# tos_help = TOSHelper(
#     bucket="ecom-govern-easyguard-sg",
#     access_key="KMDY51XQI3KV2QDNG2N7",
#     service="toutiao.tos.tosapi.service.sg1"
# )

# file_path = "/mnt/bn/ecom-ccr-dev/mlx/users/jiangjunjun.happy/llama2/7b/llama_7b_ccr"
# for filename in os.listdir(file_path):
#     print(filename)
#     input_path = os.path.join(file_path, filename)
#     tos_help.upload_model_to_tos(
#         input_path,
#         filename=filename,
#         directory="llama_7b_ccr",
#         force_overwrite=True,
#     )


# 从TOS下载模型
tos_help = TOSHelper(
    bucket="ecom-govern-easyguard-zh",
    access_key="SHZ0CK8T8963R1AVC3WT",
    service="toutiao.tos.tosapi",
)
for item in tos_help.list_files(directory="llama_7b_ccr"):
    print(item)
    tos_help.download_model_from_tos(
        filename=item,
        output_path=item,
    )
