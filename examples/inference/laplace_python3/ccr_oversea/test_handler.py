import logging

import torch
from handler_hf import EndpointHandler as HFEndpointHandler

# from handler_quant_xperf import EndpointHandler as XperfEndpointHandler
# from handler_vllm import EndpointHandler as VllmEndpointHandler

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


prompts = [
    ' Teks tersebut berasal dari laporan komoditas dari platform belanja online: "Gambar erotis" Bagaimana Anda akan memberi label pada komentar? Label: ',  # noqa: E501
    # produk berbahaya
    ' ข้อความนี้มาจากรายงานเกี่ยวกับสินค้าจากแพลตฟอร์มการช็อปปิ้งออนไลน์: " ยาไม่มีทะเบียน อันตรายต่อตังสัตว์" คุณจะติดป้ายกำกับความคิดเห็นอย่างไร? ป้ายกำกับ: ',  # noqa: E501
    # คุณสมบัติอื่น, ผลิตภัณฑ์อันตราย
    ' Văn bản xuất phát từ một báo cáo về hàng hóa từ một nền tảng mua sắm trực tuyến: "Hsnnd" Làm thế nào bạn sẽ dán nhãn bình luận? Nhãn: ',  # noqa: E501
    # không vấn đề
]
request = {"text": [prompt.encode(encoding="UTF-8") for prompt in prompts]}


hf_ep = HFEndpointHandler()
outputs = hf_ep(request)
for output in outputs["generated"]:
    logger.info(output.decode(encoding="UTF-8"))
logger.info("=" * 100)
torch.cuda.empty_cache()

# vllm_ep = VllmEndpointHandler()
# outputs = vllm_ep(request)
# for output in outputs["generated"]:
#     logger.info(output.decode(encoding="UTF-8"))
# logger.info("=" * 100)
# torch.cuda.empty_cache()

# xperf_ep = XperfEndpointHandler()
# outputs = xperf_ep(request)
# for output in outputs["generated"]:
#     logger.info(output.decode(encoding="UTF-8"))
# logger.info("=" * 100)
# torch.cuda.empty_cache()
