# 关于评测脚本的含义，可以咨询jiangjunjun.happy@bytedance.com或者hewenyao@bytedance.com
import glob
import json
import sys

import pandas as pd
from Levenshtein import distance

label_dict = {
    0: "ProductIssue--Qualification&Marking--Release&ExpirationDateIssue",
    1: "ProductIssue--Qualification&Marking--MissingGuidebook",
    2: "ProductIssue--Qualification&Marking--OtherQualification&MarkingMissing",
    3: "ProductIssue--PersonalInjury--PhysicalInjury",
    4: "ProductIssue--PersonalInjury--ChemicalInjury",
    5: "ProductIssue--PersonalInjury--OtherInjury",
    6: "ProductIssue--ProductQuality--MaterialIssues/FabricIssues",
    7: "ProductIssue--ProductQuality--PoorTaste/Effect",
    8: "ProductIssue--ProductQuality--UnsuitableSize",
    9: "ProductIssue--ProductQuality--PoorDurability",
    10: "ProductIssue--ProductQuality--UnpleasantSmell",
    11: "ProductIssue--ProductQuality--ForeignBody",
    12: "ProductIssue--ProductQuality--Rough/FlawedWorkmanship",
    13: "ProductIssue--ProductQuality--Rotted/Mouldy/Spoiled",
    14: "ProductIssue--ProductQuality--Dirty/Leakage",
    15: "ProductIssue--UserPerceptionProblem--LowCostPerformance",
    16: "ProductIssue--UserPerceptionProblem--NotRecommend/WillNotRepurchase",
    17: "ProductIssue--UserPerceptionProblem--SecondHandGoodsSuspect",
    18: "ProductIssue--UserPerceptionProblem--OtherQualityrelatedUserPerceptionProblem",
    19: "ProductIssue--Functionality--ConnectionIssue",
    20: "ProductIssue--Functionality--LossOfFunction",
    21: "ProductIssue--Functionality--Can'tTurnon/off",
    22: "ProductIssue--Functionality--ChargeFault",
    23: "ProductIssue--Functionality--OtherIssuesAffectingUsage",
    24: "ProductIssue--FalseAdvertising--MisleadingFunctionAndEffect",
    25: "ProductIssue--FalseAdvertising--MaterialCounterfeiting",
    26: "ProductIssue--NAD--InconsistentSize",
    27: "ProductIssue--NAD--InconsistentColor",
    28: "ProductIssue--NAD--InconsistentDesign",
    29: "ProductIssue--NAD--InconsistentNetWeight&Volume",
    30: "ProductIssue--NAD--OtherInconsistent",
    31: "ProductIssue--PriceComplain--Expensivethanotherplatform",
    32: "ProductIssue--PriceComplain--Expensivethanothershop",
    33: "ProductIssue--PriceComplain--PriceCutComplaint",
    34: "ProductIssue--PriceComplain--PriceIncreaseComplaint",
    35: "ProductIssue--IPR--IPR",
    36: "ProductIssue--ComplianceRisk--ProhibitedProducts",
    37: "ServiceIssue--Shipping--ForcedDelivery",
    38: "ServiceIssue--Shipping--SlowDelivery",
    39: "ServiceIssue--Shipping--WrongColor/Size/Design",
    40: "ServiceIssue--Shipping--WrongItem",
    41: "ServiceIssue--Shipping--EmptyPackage",
    42: "ServiceIssue--Shipping--MissingPiecesOrComponents",
    43: "ServiceIssue--Refundissue--SellerRejectsRefundRequest",
    44: "ServiceIssue--Refundissue--ExchangeProblem",
    45: "ServiceIssue--Refundissue--RefundProcessingIssues",
    46: "ServiceIssue--Refundissue--RefundCan'tApply",
    47: "ServiceIssue--Refundissue--NotRefundAsNegotiation",
    48: "ServiceIssue--Refundissue--ReshipmentProblem",
    49: "ServiceIssue--IllegalMarketing--RedirectTraffic",
    50: "ServiceIssue--IllegalMarketing--Conductpostivereview",
    51: "ServiceIssue--IllegalMarketing--SuspectedFraud",
    52: "ServiceIssue--IllegalMarketing--CancelOrderWithoutNegotiation",
    53: "ServiceIssue--IllegalMarketing--ExtraCharge",
    54: "ServiceIssue--IllegalMarketing--BundleSale",
    55: "ServiceIssue--ServiceCommunicate--BadAttitude",
    56: "ServiceIssue--ServiceCommunicate--NoServiceResponse",
    57: "LogisticsIssue--LogisticsSLA--SlowTimeliness/InformationLag",
    58: "LogisticsIssue--LogisticsDelivery--WrongAddress",
    59: "LogisticsIssue--LogisticsDelivery--RefuseToDeliver",
    60: "LogisticsIssue--LogisticsDelivery--NotReceived",
    61: "LogisticsIssue--LogisticsService--BadDeliveryAttitude",
    62: "LogisticsIssue--LogisticsService--CourierLostContact",
    63: "LogisticsIssue--LogisticsService--Package&DamageIssues",
    64: "LogisticsIssue--LogisticsService--ExpensiveFreight",
    65: "LogisticsIssue--LogisticsService--OtherLogisticsServiceComplaint",
    66: "LogisticsIssue--BackwardLogistics--ReturnPickUp",
    67: "LogisticsIssue--BackwardLogistics--ReturnLogisticsSLA",
    68: "LogisticsIssue--BackwardLogistics--OtherReturnLogisticsProblems",
    69: "NeutralReview",
    70: "PositiveReview",
    71: "ProductIssue--ComplianceRisk--UnsupportedProducts",
}

label_dict_simpl_GB = [
    "expiration date issue",
    "missing guidebook",
    "other qualification",
    "physical injury",
    "chemical injury",
    "other injury",
    "material or fabric issue",
    "poor taste or effect",
    "unsuitable size",
    "poor durability",
    "unpleasant smell",
    "foreign body",
    "rough or flawed workmanship",
    "rotted or mouldy or spoiled",
    "product dirty or leakage",
    "low cost performance",
    "not recommend",
    "second hand goods",
    "user perception problem",
    "connection issue",
    "partially work",
    "can not turn on or off",
    "charge fault",
    "do not work",
    "exaggerated efficacy",
    "material fraud",
    "size do not match advertised",
    "color do not match advertised",
    "style do not match advertised",
    "weight do not match advertised",
    "do not match advertised",
    "expensive than other platform",
    "expensive than other shop",
    "price cut",
    "price increase",
    "fake and pirated copycats",
    "dangerous products",
    "forced delivery",
    "slow delivery",
    "sent wrong color size",
    "sent wrong style",
    "empty package",
    "missing pieces or components",
    "reject refund",
    "exchange problem",
    "slow refund",
    "unable to request refund",
    "not refund as negotiation",
    "reshipment problem",
    "redirect traffic",
    "conduct postive review",
    "suspected fraud",
    "cancel order without negotiation",
    "extra charge",
    "bundle sale",
    "bad attitude",
    "no service response",
    "slow logistics time",
    "delivery to wrong address",
    "refuse to deliver",
    "not received",
    "bad delivery attitude",
    "courier lost contact",
    "package damage issue",
    "expensive freight",
    "other Logistics service problem",
    "return pick up",
    "return pick up slow",
    "other return logistics problem",
    "no problem",
    "positive",
    "restricted products",
]

label_dict_simpl_ID = [
    "masalah tanggal kedaluwarsa",
    "buku panduan hilang",
    "kualifikasi lain",
    "cedera fisik",
    "cedera kimia",
    "cedera lain",
    "masalah bahan atau kain",
    "rasa atau efek buruk",
    "ukuran tidak sesuai",
    "daya tahan yang buruk",
    "bau tidak sedap",
    "lembaga asing",
    "pengerjaan kasar atau cacat",
    "busuk atau berjamur atau rusak",
    "produk kotor atau bocor",
    "kinerja biaya rendah",
    "tidak merekomendasikan",
    "barang bekas",
    "masalah persepsi pengguna",
    "masalah koneksi",
    "sebagian bekerja",
    "tidak bisa hidup atau mati",
    "kesalahan muatan",
    "tidak bekerja",
    "kemanjuran berlebihan",
    "penipuan material",
    "ukuran tidak sesuai dengan yang diiklankan",
    "warna tidak sesuai dengan yang diiklankan",
    "gaya tidak cocok dengan yang diiklankan",
    "berat tidak sesuai dengan yang diiklankan",
    "tidak sesuai dengan yang diiklankan",
    "mahal dibandingkan platform lain",
    "mahal dari toko lain",
    "potongan harga",
    "harga naik",
    "peniru palsu dan bajakan",
    "produk berbahaya",
    "pengiriman paksa",
    "pengiriman lambat",
    "salah kirim ukuran warna",
    "salah mengirim gaya",
    "paket kosong",
    "bagian atau komponen yang hilang",
    "menolak pengembalian dana",
    "masalah pertukaran",
    "pengembalian dana lambat",
    "tidak dapat meminta pengembalian dana",
    "tidak pengembalian dana sebagai negosiasi",
    "masalah pengiriman ulang",
    "mengalihkan lalu lintas",
    "melakukan tinjauan positif",
    "dugaan penipuan",
    "batalkan pesanan tanpa negosiasi",
    "biaya tambahan",
    "penjualan bundel",
    "sikap buruk",
    "tidak ada tanggapan layanan",
    "waktu logistik lambat",
    "pengiriman ke alamat yang salah",
    "menolak pengiriman",
    "tidak diterima",
    "sikap pengiriman yang buruk",
    "kurir kehilangan kontak",
    "masalah kerusakan paket",
    "angkutan mahal",
    "masalah layanan Logistik lainnya",
    "penjemputan kembali",
    "kembali mengambil lambat",
    "masalah logistik pengembalian lainnya",
    "Tidak masalah",
    "positif",
    "produk yang dibatasi",
]

label_dict_simpl_MY = [
    "isu tarikh tamat tempoh",
    "buku panduan hilang",
    "kelayakan lain",
    "kecederaan fizikal",
    "kecederaan kimia",
    "kecederaan lain",
    "isu bahan atau fabrik",
    "rasa atau kesan buruk",
    "saiz tidak sesuai",
    "ketahanan lemah",
    "bau yang tidak menyenangkan",
    "badan asing",
    "kemahiran kasar atau cacat",
    "reput atau berkulat atau rosak",
    "produk kotor atau bocor",
    "prestasi kos rendah",
    "tidak mengesyorkan",
    "barang terpakai",
    "masalah persepsi pengguna",
    "isu sambungan",
    "sebahagian bekerja",
    "tidak boleh menghidupkan atau mematikan",
    "caj kesalahan",
    "tidak berfungsi",
    "keberkesanan yang berlebihan",
    "penipuan material",
    "saiz tidak sepadan dengan yang diiklankan",
    "warna tidak sepadan dengan yang diiklankan",
    "gaya tidak sepadan yang diiklankan",
    "berat tidak sepadan yang diiklankan",
    "tidak sepadan dengan yang diiklankan",
    "mahal daripada platform lain",
    "mahal daripada kedai lain",
    "potongan harga",
    "kenaikan harga",
    "peniru palsu dan cetak rompak",
    "produk berbahaya",
    "penghantaran paksa",
    "penghantaran lambat",
    "salah hantar saiz warna",
    "salah hantar gaya",
    "bungkusan kosong",
    "kepingan atau komponen yang hilang",
    "tolak bayaran balik",
    "masalah pertukaran",
    "bayaran balik lambat",
    "tidak dapat meminta bayaran balik",
    "tidak membayar balik sebagai rundingan",
    "masalah penghantaran semula",
    "ubah hala lalu lintas",
    "menjalankan semakan postif",
    "disyaki penipuan",
    "batalkan pesanan tanpa rundingan",
    "caj tambahan",
    "jualan bundle",
    "perangai buruk",
    "tiada maklum balas perkhidmatan",
    "masa logistik yang perlahan",
    "penghantaran ke alamat yang salah",
    "enggan menyampaikan",
    "tidak diterima",
    "sikap penyampaian yang buruk",
    "courier lost contact",
    "isu kerosakan pakej",
    "pengangkutan mahal",
    "masalah perkhidmatan Logistik lain",
    "jemput balik",
    "jemput balik lambat",
    "masalah logistik pemulangan lain",
    "tiada masalah",
    "positif",
    "produk terhad",
]

label_dict_simpl_PH = [
    "isyu sa petsa ng pag-expire",
    "nawawalang guidebook",
    "ibang kwalipikasyon",
    "pisikal na pinsala",
    "pinsala sa kemikal",
    "ibang pinsala",
    "isyu sa materyal o tela",
    "mahinang lasa o epekto",
    "hindi angkop na sukat",
    "mahinang tibay",
    "hindi kanais-nais na amoy",
    "katawang dayuhan",
    "magaspang o may depektong pagkakagawa",
    "bulok o inaamag o sira",
    "marumi o tumutulo ang produkto",
    "pagganap ng mababang gastos",
    "hindi inirerekomenda",
    "segunda-manong gamit",
    "problema sa pang-unawa ng gumagamit",
    "isyu sa koneksyon",
    "bahagyang gumagana",
    "hindi maaaring i-on o i-off",
    "sisingilin ang kasalanan",
    "huwag magtrabaho",
    "pinalaking bisa",
    "materyal na pandaraya",
    "hindi tugma ang laki sa na-advertise",
    "hindi tugma ang kulay sa na-advertise",
    "estilo ay hindi tumutugma sa ina-advertise",
    "hindi tugma ang timbang sa na-advertise",
    "hindi tumugma sa ina-advertise",
    "mahal kaysa sa ibang platform",
    "mahal kaysa sa ibang tindahan",
    "pagbawas ng presyo",
    "pagtaas ng presyo",
    "pekeng at pirated copycats",
    "mga mapanganib na produkto",
    "sapilitang paghahatid",
    "mabagal na paghahatid",
    "nagpadala ng maling laki ng kulay",
    "nagpadala ng maling istilo",
    "walang laman na pakete",
    "nawawalang piraso o bahagi",
    "tanggihan ang refund",
    "problema sa palitan",
    "mabagal na refund",
    "hindi makahiling ng refund",
    "hindi refund bilang negosasyon",
    "problema sa muling pagpapadala",
    "pag-redirect ng trapiko",
    "magsagawa ng postive review",
    "hinihinalang pandaraya",
    "kanselahin ang order nang walang negosasyon",
    "dagdag na bayad",
    "bundle sale",
    "masamang ugali",
    "walang tugon sa serbisyo",
    "mabagal na oras ng logistik",
    "paghahatid sa maling address",
    "tumangging maghatid",
    "hindi natanggap",
    "masamang saloobin sa paghahatid",
    "courier lost contact",
    "isyu sa pagkasira ng pakete",
    "mahal na kargamento",
    "iba pang problema sa serbisyo ng Logistics",
    "return pick up",
    "pagbalik pick up mabagal",
    "iba pang problema sa return logistics",
    "walang problema",
    "positibo",
    "mga pinaghihigpitang produkto",
]

label_dict_simpl_TH = [
    "ปัญหาวันหมดอายุ",
    "หนังสือนำเที่ยวหาย",
    "คุณสมบัติอื่น",
    "การบาดเจ็บทางร่างกาย",
    "การบาดเจ็บจากสารเคมี",
    "การบาดเจ็บอื่น",
    "ปัญหาวัสดุหรือผ้า",
    "รสชาติหรือเอฟเฟกต์แย่",
    "ขนาดไม่เหมาะสม",
    "ความทนทานต่ำ",
    "กลิ่นไม่พึงประสงค์",
    "สิ่งแปลกปลอม",
    "ฝีมือหยาบหรือมีข้อบกพร่อง",
    "เน่าหรือขึ้นราหรือบูด",
    "ผลิตภัณฑ์สกปรกหรือรั่ว",
    "ประสิทธิภาพต้นทุนต่ำ",
    "ไม่แนะนำ",
    "สินค้ามือสอง",
    "ปัญหาการรับรู้ของผู้ใช้",
    "ปัญหาการเชื่อมต่อ",
    "ทำงานบางส่วน",
    "เปิดหรือปิดไม่ได้",
    "ค่าความผิดพลาด",
    "ไม่ทำงาน",
    "สรรพคุณเกินจริง",
    "การฉ้อโกงวัสดุ",
    "ขนาดไม่ตรงกับที่โฆษณา",
    "สีไม่ตรงกับที่โฆษณา",
    "รูปแบบไม่ตรงกับที่โฆษณา",
    "น้ำหนักไม่ตรงตามโฆษณา",
    "ไม่ตรงกับโฆษณา",
    "แพงกว่าแพลตฟอร์มอื่น",
    "แพงกว่าร้านอื่น",
    "ลดราคา",
    "ขึ้นราคา",
    "ของปลอมและลอกเลียนแบบ",
    "ผลิตภัณฑ์อันตราย",
    "บังคับส่ง",
    "ส่งช้า",
    "ส่งสีผิดไซส์",
    "ส่งผิดแบบ",
    "แพ็คเกจเปล่า",
    "ชิ้นส่วนหรือส่วนประกอบที่หายไป",
    "ปฏิเสธการคืนเงิน",
    "ปัญหาการแลกเปลี่ยน",
    "คืนเงินช้า",
    "ไม่สามารถขอเงินคืนได้",
    "ไม่คืนเงินเป็นการเจรจา",
    "ปัญหาการส่งสินค้า",
    "การจราจรเปลี่ยนเส้นทาง",
    "ดำเนินการตรวจสอบภายหลัง",
    "สงสัยว่าฉ้อโกง",
    "ยกเลิกคำสั่งซื้อโดยไม่มีการเจรจา",
    "ค่าใช้จ่ายเพิ่มเติม",
    "ขายเป็นชุด",
    "ทัศนคติที่ไม่ดี",
    "ไม่มีการตอบสนองการบริการ",
    "เวลาโลจิสติกช้า",
    "จัดส่งผิดที่อยู่",
    "ปฏิเสธที่จะส่งมอบ",
    "ไม่ได้รับ",
    "ทัศนคติในการจัดส่งที่ไม่ดี",
    "ผู้จัดส่งขาดการติดต่อ",
    "ปัญหาบรรจุภัณฑ์เสียหาย",
    "ค่าขนส่งแพง",
    "ปัญหาบริการโลจิสติกส์อื่นๆ",
    "รับคืน",
    "กลับมารับช้า",
    "ปัญหาโลจิสติกส่งคืนอื่นๆ",
    "ไม่มีปัญหา",
    "เชิงบวก",
    "สินค้าที่ถูกจำกัด",
]

label_dict_simpl_VN = [
    "vấn đề ngày hết hạn",
    "thiếu sách hướng dẫn",
    "bằng cấp khác",
    "chấn thương vật lý",
    "chấn thương hóa học",
    "chấn thương khác",
    "vấn đề về chất liệu hoặc vải",
    "hương vị hoặc tác dụng kém",
    "kích thước không phù hợp",
    "độ bền kém",
    "mùi khó chịu",
    "cơ thể nước ngoài",
    "tay nghề thô hoặc thiếu sót",
    "mục nát hoặc mốc meo hoặc hư hỏng",
    "sản phẩm bẩn hoặc rò rỉ",
    "hiệu suất chi phí thấp",
    "không khuyến nghị",
    "hàng cũ",
    "vấn đề nhận thức của người dùng",
    "sự cố kết nối",
    "làm việc một phần",
    "không thể bật hoặc tắt",
    "lỗi sạc",
    "đừng làm việc",
    "hiệu quả phóng đại",
    "gian lận vật chất",
    "kích thước không phù hợp với quảng cáo",
    "màu sắc không phù hợp với quảng cáo",
    "phong cách không phù hợp với quảng cáo",
    "trọng lượng không phù hợp với quảng cáo",
    "không khớp với quảng cáo",
    "đắt hơn nền tảng khác",
    "đắt hơn cửa hàng khác",
    "giảm giá",
    "tăng giá",
    "bản sao giả mạo và vi phạm bản quyền",
    "sản phẩm nguy hiểm",
    "buộc giao hàng",
    "giao hàng chậm",
    "gửi sai kích thước màu sắc",
    "gửi sai phong cách",
    "gói rỗng",
    "thiếu mảnh hoặc thành phần",
    "từ chối hoàn tiền",
    "vấn đề trao đổi",
    "hoàn tiền chậm",
    "không thể yêu cầu hoàn tiền",
    "không hoàn trả như thương lượng",
    "vấn đề tái vận chuyển",
    "chuyển hướng lưu lượng truy cập",
    "tiến hành đánh giá tích cực",
    "nghi ngờ gian lận",
    "hủy đơn hàng mà không thương lượng",
    "phí phụ thêm",
    "bán theo gói",
    "Thái độ xấu",
    "không có phản hồi dịch vụ",
    "thời gian hậu cần chậm",
    "giao nhầm địa chỉ",
    "từ chối giao hàng",
    "không nhận",
    "thái độ giao hàng không tốt",
    "chuyển phát nhanh bị mất liên lạc",
    "vấn đề hư hỏng gói hàng",
    "vận chuyển hàng hóa đắt tiền",
    "vấn đề dịch vụ hậu cần khác",
    "nhận hàng trả lại",
    "trả khách chậm",
    "vấn đề hậu cần trả lại khác",
    "Không vấn đề",
    "tích cực",
    "sản phẩm hạn chế",
]

label_dict_simpl = {
    "GB": label_dict_simpl_GB,
    "ID": label_dict_simpl_ID,
    "MY": label_dict_simpl_MY,
    "PH": label_dict_simpl_PH,
    "TH": label_dict_simpl_TH,
    "VN": label_dict_simpl_VN,
}

label_idx = {v: k for k, v in label_dict.items()}

filter_index = [54, 1, 66, 62, 59, 33, 67, 47, 27, 20, 21, 58, 61, 11, 19]

filter_labels = [label_dict[item] for item in filter_index]
label_dict_filter = {k: v for k, v in zip(filter_index, filter_labels)}
# label_dict_filter = {}


label_levl2 = {
    # == product ==
    2: [0, 1, 2],
    5: [3, 4, 5],
    18: [15, 16, 17, 18],
    23: [19, 20, 21, 22, 23],
    30: [26, 27, 28, 29, 30],
    65: [61, 62, 63, 64, 65],
    68: [66, 67, 68],
}

label_veb = {}
for country, name_list in label_dict_simpl.items():
    for i, name in enumerate(name_list):
        standard_name = label_dict[i]
        if standard_name not in label_veb:
            label_veb[standard_name] = [name]
        else:
            names = label_veb[standard_name]
            names.append(name)
            label_veb[standard_name] = names


# handler output
def read_model_output_files(path, ori_file, country=None):
    files = glob.glob(path)
    rec = []

    for file in files:
        df = {"id": [], "output": [], "vllm": []}
        f = open(file, "r")
        line = f.readline()
        while len(line) > 0:
            tokens = json.loads(line)
            df["id"].append(tokens["idx"])
            df["output"].append(tokens["answer"])
            df["vllm"].append(tokens["vllm_answer"])
            line = f.readline()

        f.close()
        df = pd.DataFrame(df).drop_duplicates(["id"], keep="last")
        # display(df.head(3))
        rec.append(df)

    df = pd.concat(rec).drop_duplicates(["id"], keep="first")

    # 会多补几个给gpu凑batch
    df_sort = {int(df.iloc[i, 0]): df.iloc[i, 1] for i in range(len(df))}
    df_sort_vlm = {int(df.iloc[i, 0]): df.iloc[i, 2] for i in range(len(df))}

    ground_truth_df = pd.read_parquet(ori_file)
    ground_truth_df = ground_truth_df.rename(columns={"answer": "ground truth answer"})

    output_list, output_list_vlm = [], []
    for i in ground_truth_df["id"]:
        if i in df_sort:
            output_list.append(df_sort[i])
        else:
            output_list.append("")
        if i in df_sort_vlm:
            output_list_vlm.append(df_sort_vlm[i])
        else:
            output_list_vlm.append("")

    ground_truth_df["output"] = output_list
    ground_truth_df["vllm_output"] = output_list_vlm
    ground_truth_df = ground_truth_df.loc[
        :, ["id", "sent1", "country", "question", "ground truth answer", "output", "vllm_output"]
    ]

    if country:
        return ground_truth_df[ground_truth_df["country"] == country]
    else:
        return ground_truth_df


def check(p_list, l_list):
    acc = 0
    softacc, soft_cnt = 0, 0
    batch_size = len(l_list)
    num_labels = len(l_list[0])
    top1acc_list = []
    for i in range(batch_size):
        if 1 not in p_list[i]:
            continue
        if p_list[i] == l_list[i]:
            acc += 1

        # == softacc ==
        id = 0
        for j in range(num_labels):
            if l_list[i][j] == 1 and j not in label_dict_filter:
                soft_cnt += 1
                id = 1
                for k in range(num_labels):
                    if p_list[i][k] == 1:
                        if l_list[i][k] != 1:
                            id = 0
                            break
                softacc += id
                break

        top1acc_list.append(id)

    return softacc / soft_cnt, top1acc_list


def Per_Rec(p_list, l_list):
    batch_size = len(l_list)
    num_labels = len(l_list[0])

    label_result_dict = {}
    pred_result_dict = {}
    pred_true_dict = {}
    rec_true_dict = {}
    P = {}
    R = {}
    F1 = {}
    P_merge = 0
    R_merge = 0
    F1_merge = 0

    for k, v in label_dict.items():
        label_result_dict[v] = 0
        pred_result_dict[v] = 0
        pred_true_dict[v] = 0
        rec_true_dict[v] = 0
        P[v] = 0
        R[v] = 0

    for i in range(batch_size):
        for j in range(num_labels):
            if p_list[i][j] == 1:
                pred_result_dict[label_dict[j]] += 1
                if j in label_levl2:  # todo level2
                    for idx in label_levl2[j]:
                        if l_list[i][idx] == 1:
                            pred_true_dict[label_dict[j]] += 1
                            break
                elif l_list[i][j] == 1:
                    pred_true_dict[label_dict[j]] += 1

    c1, c2 = 0, 0
    for k, v in pred_true_dict.items():
        P[k] = v / (pred_result_dict[k] + 0.0001)
        if k not in label_dict_filter.values():  # todo add label filter
            P_merge += P[k]

        c1 += v
        c2 += pred_result_dict[k]
    # P_micro = c1 / c2

    for i in range(batch_size):
        for j in range(num_labels):
            if l_list[i][j] == 1:
                label_result_dict[label_dict[j]] += 1
                if j in label_levl2:  # todo level2
                    for idx in label_levl2[j]:
                        if p_list[i][idx] == 1:
                            rec_true_dict[label_dict[j]] += 1
                            break
                elif p_list[i][j] == 1:
                    rec_true_dict[label_dict[j]] += 1

    c3, c4 = 0, 0
    for k, v in rec_true_dict.items():
        R[k] = v / (label_result_dict[k] + 0.0001)
        F1[k] = 2 * P[k] * R[k] / (P[k] + R[k] + 0.0001)
        if k not in label_dict_filter.values():  # todo add label filter
            R_merge += R[k]
            F1_merge += F1[k]

        c3 += v
        c4 += label_result_dict[k]
    R_micro = c3 / c4
    F_micro = 2 * R_micro * R_micro / (R_micro + R_micro + 0.0001)

    P_merge /= num_labels - len(label_dict_filter)  # todo add label filter
    R_merge /= num_labels - len(label_dict_filter)
    F1_merge /= num_labels - len(label_dict_filter)

    return P_merge, R_merge, F1_merge, P, R, F_micro


def getMaxCommonSubstr(s1, s2):
    # 求两个字符串的最长公共子串
    # 思想：建立一个二维数组，保存连续位相同与否的状态

    len_s1 = len(s1)
    len_s2 = len(s2)

    # 生成0矩阵，为方便后续计算，多加了1行1列
    # 行: (len_s1+1)
    # 列: (len_s2+1)
    record = [[0 for i in range(len_s2 + 1)] for j in range(len_s1 + 1)]

    maxNum = 0  # 最长匹配长度
    # p = 0  # 字符串匹配的终止下标

    for i in range(len_s1):
        for j in range(len_s2):
            if s1[i] == s2[j]:
                # 相同则累加
                record[i + 1][j + 1] = record[i][j] + 1

                if record[i + 1][j + 1] > maxNum:
                    maxNum = record[i + 1][j + 1]
                    # p = i  # 匹配到下标i

    # 返回 子串长度，子串
    # return maxNum, s1[p + 1 - maxNum: p + 1]
    return maxNum


def multi_compute_metrics(all_predicts_list, all_labels_list):
    # batch_size = len(all_labels_list)
    num_labels = 72

    preds_list = []
    no_matched_cases, no_matched_cases2 = 0, 0
    no_matched_flag = []
    for i, predict in enumerate(all_predicts_list):
        preds = [0 for i in range(num_labels)]
        for standard_name, names in label_veb.items():
            for name in names:
                if name in predict:
                    preds[label_idx[standard_name]] = 1

        if 1 not in preds:
            no_matched_cases += 1
            no_matched_flag.append(1)
            # == 1 select neutral ==
            # preds[-2] = 1
            # == 2 select min distince ==
            for p in predict.split(", "):
                dis_dict = {}
                for standard_name, names in label_veb.items():
                    min_dis = 100000
                    # max_common = 0
                    for name in names:
                        min_dis = min(min_dis, distance(name, p))
                        # max_common = max(max_common, getMaxCommonSubstr(name, predict))
                    # dis_dict[standard_name] = (min_dis, max_common)
                    dis_dict[standard_name] = min_dis
                    # dis_dict[standard_name] = max_common
                # dis_dict_sorted = sorted(dis_dict.items(), key=lambda kv: (kv[1][0], -kv[1][1]), reverse=False)
                dis_dict_sorted = sorted(dis_dict.items(), key=lambda kv: kv[1], reverse=False)
                # dis_dict_sorted = sorted(dis_dict.items(), key=lambda kv: kv[1], reverse=True)
                preds[label_idx[dis_dict_sorted[0][0]]] = 1
            if dis_dict_sorted[0][1] == dis_dict_sorted[1][1]:
                no_matched_cases2 += 1
        else:
            no_matched_flag.append(0)

        preds_list.append(preds)
    print("no_matched_cases: ", no_matched_cases, no_matched_cases2)

    labels_list = []
    for i, labels in enumerate(all_labels_list):
        label = [0 for i in range(num_labels)]
        for standard_name, names in label_veb.items():
            for l in labels.split(", "):  # noqa: E741
                if l in names:
                    label[label_idx[standard_name]] = 1

        # if 1 not in label:
        #     print ('error, not found label name!')

        labels_list.append(label)

    top1acc, top1acc_list = check(preds_list, labels_list)
    P_merge, R_merge, F1_merge, P, R, F_micro = Per_Rec(preds_list, labels_list)
    return [top1acc, P_merge, R_merge, F1_merge, P, R, top1acc_list, no_matched_flag, F_micro]


def run(model_output, ori_file, country=None):
    df = read_model_output_files(model_output, ori_file, country)

    all_labels_list = df["ground truth answer"].tolist()
    all_predicts_list = df["vllm_output"].tolist()
    res = multi_compute_metrics(all_predicts_list, all_labels_list)
    df["acc_list"] = res[6]  # prediction is correction or not
    df["fuzzy_matched_flag"] = res[7]  # is fuzzy matched flag (based on edit distance)

    print("F1_macro =", f"{res[3]:.4f}", "F1_micro =", f"{res[-1]:.4f}", "top1acc =", f"{res[0]:.4f}")

    return df  # 反回统计的df细节, 用于debug


if __name__ == "__main__":
    model_output = sys.argv[1]
    ori_file = sys.argv[2]
    print("")
    print("Overall")
    df = run(model_output, ori_file)
    print()

    for ct in list(label_dict_simpl.keys()):
        print(ct)
        run(model_output, ori_file, ct)
