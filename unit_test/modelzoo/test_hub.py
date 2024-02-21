import unittest

# test module

TEST_FLAGS = ["all"]


class TestHub(unittest.TestCase):
    def test_BaseAutoHubClass(self):
        from easyguard.modelzoo.hub import AutoHubClass

        class a(AutoHubClass):
            def __init__(
                self,
                server_name: str,
                archive_name: str,
                model_type: str,
                *args,
                **kwargs,
            ) -> None:
                super().__init__()

        # x = a("hdfs", "fashionxlm-moe-base", "mdeberta_v2_moe", region="CN")
        # x.get_file("config.json")


if __name__ == "__main__":
    unittest.main()
