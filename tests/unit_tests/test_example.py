import pytest


class TestDemo:
    def test_a(self):
        assert not 1 == 2

    def test_b(self):
        assert 3 == 3

    def test_c(self):
        assert 5 == 5


if __name__ == "__main__":
    pytest.main()
