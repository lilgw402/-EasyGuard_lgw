import unittest

# test module

TEST_FLAGS = ["all"]


class TestYaml(unittest.TestCase):
    @unittest.skipUnless("all" in TEST_FLAGS or "json2yaml" in TEST_FLAGS, "just do it")
    def test_json2yaml(self):
        from easyguard.utils.yaml_utils import YamlConfig, json2yaml

        class ModelZooYaml(YamlConfig):
            def check(self):
                ...

            def test(self):
                print("123")

        data = {"name": "djw", "indicators": [1, 2, 3, 4]}

        path = r"/mnt/bn/ecom-govern-maxiangqian/dongjunwei/EasyGuard/tests/utils/output.yaml"
        json2yaml(path, data)


if __name__ == "__main__":
    unittest.main()
