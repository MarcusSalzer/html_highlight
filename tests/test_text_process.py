import unittest
from src import text_process


class TestTextprocess(unittest.TestCase):
    def test_single_stmt(self):
        tk, ta = text_process.process_regex("x = 3")
        self.assertListEqual(["x", " ", "=", " ", "3"], tk)
        tk, ta = text_process.process_regex("3+f(y)")
        self.assertListEqual(["3", "+", "f", "(", "y", ")"], tk)
        self.assertEqual("nu", ta[0])

    def test_strs(self):
        tk, ta = text_process.process_regex("name = 'abc';")
        self.assertListEqual(["name", " ", "=", " ", "'abc'", ";"], tk)
        tk, ta = text_process.process_regex('name = "abc"')
        self.assertListEqual(["name", " ", "=", " ", '"abc"'], tk)
        # tk, ta = text_process.tokenize_plus('a "nested" string')
        # self.assertListEqual(["a", " ", '"', "nested", '"', " ", "string"], tk)

    def test_multiline(self):
        tk, ta = text_process.process_regex("name='abc'\nx=2")
        self.assertListEqual(["name", "=", "'abc'", "\n", "x", "=", "2"], tk)

        self.assertEqual("nl", ta[3])
        self.assertEqual("nu", ta[-1])

        tk, ta = text_process.process_regex("range(12):\n  print")
        self.assertListEqual(["range", "(", "12", ")", ":", "\n", "  ", "print"], tk)
        self.assertEqual("nu", ta[2])
        self.assertEqual("nl", ta[5])
        self.assertEqual("ws", ta[6])

        tk, ta = text_process.process_regex("while true{\n    do_something\n}")
        self.assertListEqual(
            ["while", " ", "true", "{", "\n", "    ", "do_something", "\n", "}"], tk
        )
        self.assertEqual("ws", ta[1])
        self.assertEqual("nl", ta[4])
        self.assertEqual("ws", ta[5])
        self.assertEqual("nl", ta[7])

    def test_number(self):
        tk, ta = text_process.process_regex("1337, 12")
        self.assertListEqual(
            ["1337", ",", " ", "12"],
            tk,
            "two ints",
        )

        tk, ta = text_process.process_regex("1337.0, 12.55")
        self.assertListEqual(
            ["1337.0", ",", " ", "12.55"],
            tk,
            "two floats",
        )

        self.assertEqual("nu", ta[0])
        self.assertEqual("ws", ta[2])
        self.assertEqual("nu", ta[3])

    def test_numbers_advanced(self):
        tk, ta = text_process.process_regex("1337usize>2i32")
        self.assertListEqual(
            ["1337usize", ">", "2i32"],
            tk,
            "annotated nums",
        )

        tk, ta = text_process.process_regex("1_000>1.0f64")
        self.assertListEqual(
            ["1_000", ">", "1.0f64"],
            tk,
            "annotated nums",
        )

        tk, ta = text_process.process_regex("1337_usize>2_i32")
        self.assertListEqual(
            ["1337_usize", ">", "2_i32"],
            tk,
            "annotated nums with underscores",
        )

        tk, ta = text_process.process_regex("1337_000 < 2_000.123")
        self.assertListEqual(
            ["1337_000", " ", "<", " ", "2_000.123"],
            tk,
            "nums with underscores",
        )

        tk, ta = text_process.process_regex("3+1i")
        self.assertListEqual(
            ["3", "+", "1i"],
            tk,
            "complex number",
        )

        tk, ta = text_process.process_regex("x3 = 3*1i")
        self.assertListEqual(
            ["x3", " ", "=", " ", "3", "*", "1i"],
            tk,
        )

        tk, ta = text_process.process_regex("22/7")
        self.assertListEqual(
            ["22", "/", "7"],
            tk,
            "division of ints",
        )

        tk, ta = text_process.process_regex("22.02/7.12")
        self.assertListEqual(
            ["22.02", "/", "7.12"],
            tk,
            "division of floats",
        )

    def test_numbers_hex(self):
        tk, ta = text_process.process_regex("0x0")
        self.assertListEqual(
            ["0x0"],
            tk,
        )

        tk, ta = text_process.process_regex("a=0xA")
        self.assertListEqual(
            ["a", "=", "0xA"],
            tk,
        )

        tk, ta = text_process.process_regex("a=0x1a2B + 33")
        self.assertListEqual(
            ["a", "=", "0x1a2B", " ", "+", " ", "33"],
            tk,
        )

        tk, ta = text_process.process_regex("a==0xdeadbeef")
        self.assertListEqual(
            ["a", "==", "0xdeadbeef"],
            tk,
        )

    def test_numbers_scientific(self):
        tk, ta = text_process.process_regex("1e0,-1e3,7e77,33.5e-12")
        self.assertListEqual(
            ["1e0", ",", "-", "1e3", ",", "7e77", ",", "33.5e-12"],
            tk,
        )
        self.assertEqual("nu", ta[3])
        self.assertEqual("nu", ta[5])
        self.assertEqual("nu", ta[7])
        self.assertEqual("nu", ta[0])

    def test_num_and_var(self):
        tk, ta = text_process.process_regex("_player = 3")
        self.assertEqual(["_player", " ", "=", " ", "3"], tk)
        self.assertNotEqual("nu", ta[0])


if __name__ == "__main__":
    unittest.main()
