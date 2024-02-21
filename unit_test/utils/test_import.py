import unittest

# test module

TEST_FLAGS = ["all"]


class TestImport(unittest.TestCase):
    @unittest.skipUnless("all" in TEST_FLAGS or "model_import" in TEST_FLAGS, "just do it")
    def test_lazy_model_import(self):
        from easyguard.utils.import_utils import lazy_model_import

        module_name = "models.bert"
        lazy_model_import(module_name)


if __name__ == "__main__":
    unittest.main()
