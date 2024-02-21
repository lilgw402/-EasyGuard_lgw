import unittest

# test module

TEST_FLAGS = ["all"]


class TestConfig(unittest.TestCase):
    @unittest.skipUnless("all" in TEST_FLAGS or "config" in TEST_FLAGS, "just do it")
    def test_config(self):
        ...


if __name__ == "__main__":
    unittest.main()
