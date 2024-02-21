import unittest

# test module

TEST_FLAGS = ["all"]


class TestHDFS(unittest.TestCase):
    @unittest.skipUnless("all" in TEST_FLAGS or "hopen" in TEST_FLAGS, "just do it")
    def test_hopen():
        from easyguard.utils.hdfs_utils import hopen

        hdfs_file = (
            "hdfs://haruna/home/byte_ecom_govern/user/yangzheming/asr_model/zh_deberta_base_l6_emd_20210720/vocab.txt"
        )
        vocab_set = set()
        with hopen(hdfs_file) as fr:
            for item in fr:
                item = str(item, encoding="utf-8")
                item = item.strip()
                vocab_set.add(item)
        for item in vocab_set:
            print(item)
            break

        is_bool = "中亚" in vocab_set
        print(is_bool)


if __name__ == "__main__":
    unittest.main()
