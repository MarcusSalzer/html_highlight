import unittest

from src.text_process import process_regex, process


class TestEasy(unittest.TestCase):
    def test_addnums(self):
        tk, ta = process_regex("1+2 = 3")
        self.assertListEqual(["1", "+", "2", " ", "=", " ", "3"], tk)
        self.assertEqual("nu", ta[0])
        self.assertEqual("nu", ta[2])
        self.assertEqual("ws", ta[3])

    def test_str_small(self):
        tk, ta = process_regex("'abc' \"xyz\"")
        self.assertListEqual(["'abc'", " ", '"xyz"'], tk)
        self.assertListEqual(["st", "ws", "st"], ta)

    def test_w_indent_sp(self):
        tk, ta = process("a\n  b = a")
        self.assertListEqual(["a", "\n", "  ", "b", " ", "=", " ", "a"], tk)
        self.assertEqual("nl", ta[1])
        self.assertEqual("id", ta[2])
        self.assertEqual("ws", ta[4])
        self.assertEqual("ws", ta[6])

    def test_w_indent_ta(self):
        tk, ta = process("a\n\tb = a")
        self.assertListEqual(["a", "\n", "\t", "b", " ", "=", " ", "a"], tk)
        self.assertEqual("nl", ta[1])
        self.assertEqual("id", ta[2])
        self.assertEqual("ws", ta[4])
        self.assertEqual("ws", ta[6])


class TestComment(unittest.TestCase):
    def test_cofl_py(self):
        tk, ta = process_regex("# add (+) two numbers\n1+1")
        self.assertListEqual(["# add (+) two numbers", "\n", "1", "+", "1"], tk)
        self.assertListEqual(["cofl", "nl", "nu"], ta[:3])

    def test_cofl_c(self):
        tk, ta = process_regex("// a comment 123 'str'\n")
        self.assertListEqual(["// a comment 123 'str'", "\n"], tk)
        self.assertListEqual(["cofl", "nl"], ta)


if __name__ == "__main__":
    unittest.main()
