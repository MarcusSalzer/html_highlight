import unittest
from src import text_process


class TestTokenize(unittest.TestCase):
    def test_single_stmt(self):
        tk, ta = text_process.tokenize_plus("x = 3")
        self.assertListEqual(["x", " ", "=", " ", "3"], tk)
        self.assertListEqual(["unk", "wsp", "unk", "wsp", "num"], ta)
        tk, ta = text_process.tokenize_plus("3+f(y)")
        self.assertListEqual(["3", "+", "f", "(", "y", ")"], tk)
        self.assertListEqual(["num", "unk", "unk", "unk", "unk", "unk"], ta)

    def test_strs(self):
        tk, ta = text_process.tokenize_plus("name = 'abc';")
        self.assertListEqual(["name", " ", "=", " ", "'", "abc", "'", ";"], tk)
        tk, ta = text_process.tokenize_plus('name = "abc"')
        self.assertListEqual(["name", " ", "=", " ", '"', "abc", '"'], tk)
        tk, ta = text_process.tokenize_plus('a "nested" string')
        self.assertListEqual(["a", " ", '"', "nested", '"', " ", "string"], tk)

    def test_multiline(self):
        tk, ta = text_process.tokenize_plus("name='abc'\nx=2")
        self.assertListEqual(["name", "=", "'", "abc", "'", "\n", "x", "=", "2"], tk)

        tk, ta = text_process.tokenize_plus("range(12):\n  print")
        self.assertListEqual(["range", "(", "12", ")", ":", "\n", "  ", "print"], tk)

        tk, ta = text_process.tokenize_plus("while true{\n    do_something\n}")
        self.assertListEqual(
            ["while", " ", "true", "{", "\n", "    ", "do_something", "\n", "}"], tk
        )

    def test_number(self):
        tk, ta = text_process.tokenize_plus("1337, 12")
        self.assertListEqual(
            ["1337", ",", " ", "12"],
            tk,
            "two ints",
        )

        tk, ta = text_process.tokenize_plus("1337.0, 12.55")
        self.assertListEqual(
            ["1337.0", ",", " ", "12.55"],
            tk,
            "two floats",
        )
        self.assertListEqual(
            ["num", "unk", "wsp", "num"],
            ta,
            "two floats",
        )

    def test_numbers_advanced(self):
        tk, ta = text_process.tokenize_plus("1337usize>2i32")
        self.assertListEqual(
            ["1337usize", ">", "2i32"],
            tk,
            "annotated nums",
        )

        tk, ta = text_process.tokenize_plus("1_000>1.0f64")
        self.assertListEqual(
            ["1_000", ">", "1.0f64"],
            tk,
            "annotated nums",
        )

        tk, ta = text_process.tokenize_plus("1337_usize>2_i32")
        self.assertListEqual(
            ["1337_usize", ">", "2_i32"],
            tk,
            "annotated nums with underscores",
        )

        tk, ta = text_process.tokenize_plus("1337_000 < 2_000.123")
        self.assertListEqual(
            ["1337_000", " ", "<", " ", "2_000.123"],
            tk,
            "nums with underscores",
        )

        tk, ta = text_process.tokenize_plus("3+1i")
        self.assertListEqual(
            ["3", "+", "1i"],
            tk,
            "complex number",
        )

        tk, ta = text_process.tokenize_plus("x3 = 3*1i")
        self.assertListEqual(
            ["x3", " ", "=", " ", "3", "*", "1i"],
            tk,
        )

        tk, ta = text_process.tokenize_plus("22/7")
        self.assertListEqual(
            ["22", "/", "7"],
            tk,
            "division of ints",
        )

        tk, ta = text_process.tokenize_plus("22.02/7.12")
        self.assertListEqual(
            ["22.02", "/", "7.12"],
            tk,
            "division of floats",
        )

    def test_numbers_hex(self):
        tk, ta = text_process.tokenize_plus("0x0")
        self.assertListEqual(
            ["0x0"],
            tk,
        )

        tk, ta = text_process.tokenize_plus("a=0xA")
        self.assertListEqual(
            ["a", "=", "0xA"],
            tk,
        )

        tk, ta = text_process.tokenize_plus("a=0x1a2B + 33")
        self.assertListEqual(
            ["a", "=", "0x1a2B", " ", "+", " ", "33"],
            tk,
        )

        tk, ta = text_process.tokenize_plus("a==0xdeadbeef")
        self.assertListEqual(
            ["a", "=", "=", "0xdeadbeef"],
            tk,
        )

    def test_numbers_scientific(self):
        tk, ta = text_process.tokenize_plus("1e0,-1e3,7e77,33.5e-12")
        self.assertListEqual(
            ["1e0", ",", "-", "1e3", ",", "7e77", ",", "33.5e-12"],
            tk,
        )
        self.assertListEqual(
            ["num", "unk", "unk", "num", "unk", "num", "unk", "num"],
            ta,
        )


if __name__ == "__main__":
    unittest.main()
