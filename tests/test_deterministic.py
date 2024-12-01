import unittest

from src.text_process import process_regex


class TestEasy(unittest.TestCase):
    def test_addnums(self):
        tk, ta = process_regex("1+2 = 3")
        self.assertListEqual(["1", "+", "2", " ", "=", " ", "3"], tk)
        self.assertListEqual(["nu", "op", "nu", "ws", "opas", "ws", "nu"], ta)

    def test_str_small(self):
        tk, ta = process_regex("'abc' \"xyz\"")
        self.assertListEqual(["'abc'", " ", '"xyz"'], tk)
        self.assertListEqual(["st", "ws", "st"], ta)


class TestComment(unittest.TestCase):
    def test_cofl_py(self):
        tk, ta = process_regex("# add (+) two numbers\n1+1")
        self.assertListEqual(["# add (+) two numbers", "\n", "1", "+", "1"], tk)
        self.assertListEqual(["cofl", "nl", "nu", "op", "nu"], ta)

    def test_cofl_c(self):
        tk, ta = process_regex("// a comment 123 'str'\n")
        self.assertListEqual(["// a comment 123 'str'", "\n"], tk)
        self.assertListEqual(["cofl", "nl"], ta)


if __name__ == "__main__":
    unittest.main()
