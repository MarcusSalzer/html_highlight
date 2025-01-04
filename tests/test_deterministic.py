import unittest

from src.text_process import process


class TestEasy(unittest.TestCase):
    def test_addnums(self):
        tk, ta = process("1+2 = 3")
        self.assertListEqual(["1", "+", "2", " ", "=", " ", "3"], tk)
        self.assertEqual("nu", ta[0])
        self.assertEqual("nu", ta[2])
        self.assertEqual("ws", ta[3])

    def test_str_small(self):
        tk, ta = process("'abc' \"xyz\"")
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


# class TestOp(unittest.TestCase):
#     def test_intdiv_py(self):
#         tk,ta =


class TestComment(unittest.TestCase):
    def test_cofl_py(self):
        tk, ta = process("# add (+) two numbers\n1+1")
        self.assertListEqual(["# add (+) two numbers", "\n", "1", "+", "1"], tk)
        self.assertListEqual(["cofl", "nl", "nu"], ta[:3])

    def test_cofl_c(self):
        tk, ta = process("// a comment 123 'str'\n14")
        self.assertListEqual(["// a comment 123 'str'", "\n", "14"], tk)
        self.assertListEqual(["cofl", "nl", "nu"], ta)

    def test_coil_py(self):
        tk, ta = process("x=3 # a number (3)\n14")
        self.assertListEqual(["x", "=", "3", " ", "# a number (3)", "\n", "14"], tk)
        self.assertEqual("coil", ta[4])

    def test_coil_c(self):
        tk, ta = process("x=3 // a number (3)")
        self.assertListEqual(["x", "=", "3", " ", "// a number (3)"], tk)
        self.assertEqual("coil", ta[4])

    def test_coml_php(self):
        tk, ta = process("/**\n * Store user.\n */\npublic function store")
        self.assertListEqual(
            ["/**\n * Store user.\n */", "\n", "public", " ", "function", " ", "store"],
            tk,
        )
        self.assertEqual("coml", ta[0])
        self.assertEqual("nl", ta[1])

    def test_2coml_php(self):
        tk, ta = process("\n/**\n * Store user.\n */\npublic function store" * 2)
        self.assertListEqual(
            [
                "\n",
                "/**\n * Store user.\n */",
                "\n",
                "public",
                " ",
                "function",
                " ",
                "store",
            ]
            * 2,
            tk,
        )
        self.assertEqual("nl", ta[0])
        self.assertEqual("coml", ta[1])
        self.assertEqual("nl", ta[2])
        self.assertEqual("nl", ta[8])
        self.assertEqual("coml", ta[9])


if __name__ == "__main__":
    unittest.main()
