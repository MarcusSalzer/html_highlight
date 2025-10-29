import unittest

from src import text_process
from src.text_process import process_regex


class TestCleanUp(unittest.TestCase):
    def test_2lines(self):
        t = "a\nb\n"
        c = text_process.clean_text(t)
        self.assertEqual("a\nb", c)

    def test_2lines_br(self):
        t = "a\n\n  b\n"
        c = text_process.clean_text(t)
        self.assertEqual("a\n\n  b", c)

    def test_trail_nl(self):
        t = "x = 3\ny = 2\n"
        c = text_process.clean_text(t)
        self.assertEqual("x = 3\ny = 2", c)

    def test_trail_2nl(self):
        t = "x = 3\ny = 2\n\n"
        c = text_process.clean_text(t)
        self.assertEqual("x = 3\ny = 2", c)

    def test_trail_spt(self):
        t = "x = 3 \ny = 2\t\n\nprint(x+y)"
        c = text_process.clean_text(t)
        self.assertEqual("x = 3\ny = 2\n\nprint(x+y)", c)


class TestInitialRegex(unittest.TestCase):
    def test_single_stmt(self):
        tk, ta = process_regex("x = 3")
        self.assertListEqual(["x", " ", "=", " ", "3"], tk)
        tk, ta = process_regex("3+f(y)")
        self.assertListEqual(["3", "+", "f", "(", "y", ")"], tk)
        self.assertEqual("nu", ta[0])

    def test_strs(self):
        tk, ta = process_regex("name = 'abc';")
        self.assertListEqual(["name", " ", "=", " ", "'abc'", ";"], tk)
        self.assertEqual("st", ta[4])
        tk, ta = process_regex('name = "abc"')
        self.assertEqual("st", ta[4])
        self.assertListEqual(["name", " ", "=", " ", '"abc"'], tk)

    def test_str_phpkey(self):
        t = "\n    'type' => $exData['type']"
        tk, ta = process_regex(t)
        self.assertListEqual(
            ["\n", "    ", "'type'", " ", "=>", " ", "$exData", "[", "'type'", "]"],
            tk,
        )
        self.assertEqual("st", ta[2])
        self.assertEqual("st", ta[8])

    def test_lambda_arrow_big(self):
        tk, ta = process_regex("(m) => parse(m)")
        self.assertListEqual(
            ["(", "m", ")", " ", "=>", " ", "parse", "(", "m", ")"],
            tk,
        )

    def test_lambda_arrow_small(self):
        tk, ta = process_regex("(m) -> parse(m)")
        self.assertListEqual(
            ["(", "m", ")", " ", "->", " ", "parse", "(", "m", ")"],
            tk,
        )

    def test_multiline(self):
        tk, ta = process_regex("name='abc'\nx=2")
        self.assertListEqual(["name", "=", "'abc'", "\n", "x", "=", "2"], tk)

        self.assertEqual("nl", ta[3])
        self.assertEqual("nu", ta[-1])

    def test_multiline_id2(self):
        tk, ta = text_process.process("range(12):\n  print")
        self.assertListEqual(["range", "(", "12", ")", ":", "\n", "  ", "print"], tk)
        self.assertEqual("nu", ta[2])
        self.assertEqual("nl", ta[5])
        self.assertEqual("id", ta[6])

    def test_multiline_id4(self):
        tk, ta = text_process.process("while true{\n    do_something\n}")
        self.assertListEqual(
            ["while", " ", "true", "{", "\n", "    ", "do_something", "\n", "}"], tk
        )
        self.assertEqual("ws", ta[1])
        self.assertEqual("nl", ta[4])
        self.assertEqual("id", ta[5])
        self.assertEqual("nl", ta[7])

    def test_more_id(self):
        tk, ta = text_process.process("1\n  2\n    3")
        self.assertListEqual(["1", "\n", "  ", "2", "\n", "    ", "3"], tk)
        self.assertListEqual(["nu", "nl", "id", "nu", "nl", "id", "nu"], ta)

    def test_number(self):
        tk, ta = process_regex("1337, 12")
        self.assertListEqual(
            ["1337", ",", " ", "12"],
            tk,
            "two ints",
        )

        tk, ta = process_regex("1337.0, 12.55")
        self.assertListEqual(
            ["1337.0", ",", " ", "12.55"],
            tk,
            "two floats",
        )

        self.assertEqual("nu", ta[0])
        self.assertEqual("ws", ta[2])
        self.assertEqual("nu", ta[3])

    def test_numbers_advanced(self):
        tk, ta = process_regex("1337usize>2i32")
        self.assertListEqual(
            ["1337usize", ">", "2i32"],
            tk,
            "annotated nums",
        )

        tk, ta = process_regex("1_000>1.0f64")
        self.assertListEqual(
            ["1_000", ">", "1.0f64"],
            tk,
            "annotated nums",
        )

        tk, ta = process_regex("1337_usize>2_i32")
        self.assertListEqual(
            ["1337_usize", ">", "2_i32"],
            tk,
            "annotated nums with underscores",
        )

        tk, ta = process_regex("1337_000 < 2_000.123")
        self.assertListEqual(
            ["1337_000", " ", "<", " ", "2_000.123"],
            tk,
            "nums with underscores",
        )

        tk, ta = process_regex("3+1i")
        self.assertListEqual(
            ["3", "+", "1i"],
            tk,
            "complex number",
        )
        tk, ta = process_regex("1.5j")
        self.assertListEqual(
            ["1.5j"],
            tk,
            "complex decimal number",
        )
        self.assertEqual("nu", ta[0])

        tk, ta = process_regex("1f 1.333f")
        self.assertListEqual(
            ["1f", " ", "1.333f"],
            tk,
            "float literal",
        )
        self.assertEqual("nu", ta[0])
        self.assertEqual("nu", ta[2])

        tk, ta = process_regex("1.5j")
        self.assertListEqual(
            ["1.5j"],
            tk,
            "complex decimal number",
        )

        tk, ta = process_regex("x3 = 3*1i")
        self.assertListEqual(
            ["x3", " ", "=", " ", "3", "*", "1i"],
            tk,
        )

        tk, ta = process_regex("22/7")
        self.assertListEqual(
            ["22", "/", "7"],
            tk,
            "division of ints",
        )

        tk, ta = process_regex("22.02/7.12")
        self.assertListEqual(
            ["22.02", "/", "7.12"],
            tk,
            "division of floats",
        )

    def test_numbers_hex(self):
        tk, ta = process_regex("0x0")
        self.assertListEqual(
            ["0x0"],
            tk,
        )

        tk, ta = process_regex("a=0xA")
        self.assertListEqual(
            ["a", "=", "0xA"],
            tk,
        )
        self.assertEqual("nu", ta[2])

        tk, ta = process_regex("a=0x1a2B + 33")
        self.assertListEqual(
            ["a", "=", "0x1a2B", " ", "+", " ", "33"],
            tk,
        )

        tk, ta = process_regex("a==0xdeadbeef")
        self.assertListEqual(
            ["a", "==", "0xdeadbeef"],
            tk,
        )

    def test_numbers_scientific(self):
        tk, ta = process_regex("1e0,-1e3,7e77,33.5e-12,1e-10")
        self.assertListEqual(
            ["1e0", ",", "-", "1e3", ",", "7e77", ",", "33.5e-12", ",", "1e-10"],
            tk,
        )
        self.assertEqual("nu", ta[0])
        self.assertEqual("nu", ta[3])
        self.assertEqual("nu", ta[5])
        self.assertEqual("nu", ta[7])

    def test_num_and_var(self):
        tk, ta = process_regex("_player = 3")
        self.assertEqual(["_player", " ", "=", " ", "3"], tk)
        self.assertNotEqual("nu", ta[0])

    def test_num_range(self):
        tk, ta = process_regex("1...9")
        self.assertListEqual(["1", "...", "9"], tk)
        self.assertListEqual(["nu", "sy", "nu"], ta)

        tk, ta = process_regex("1..9")
        self.assertListEqual(["1", "..", "9"], tk)
        self.assertListEqual(["nu", "sy", "nu"], ta)

    def test_varname_w_num(self):
        tk, ta = process_regex("a12 xy_3 9")
        self.assertListEqual(["a12", " ", "xy_3", " ", "9"], tk)

    def test_rust_lifetime(self):
        t = "&'a str"
        tk, ta = process_regex(t)
        self.assertListEqual(["&", "'a", " ", "str"], tk)
        self.assertEqual("an", ta[1])

    def test_rust_lifetime_fun(self):
        t = "fn longest<'a>(x: &'a str)"
        tk, ta = process_regex(t)
        self.assertListEqual(
            [
                "fn",
                " ",
                "longest",
                "<",
                "'a",
                ">",
                "(",
                "x",
                ":",
                " ",
                "&",
                "'a",
                " ",
                "str",
                ")",
            ],
            tk,
        )

    def test_rust_lifetime_two(self):
        tk, ta = process_regex("fn foo<'a, 'b>")
        self.assertListEqual(["fn", " ", "foo", "<", "'a", ",", " ", "'b", ">"], tk)
        self.assertEqual("an", ta[4])
        self.assertEqual("an", ta[7])

    def test_rust_lifetime_tworef(self):
        tk, ta = process_regex("x: &'a str, y: &'b str")
        self.assertListEqual(
            [
                "x",
                ":",
                " ",
                "&",
                "'a",
                " ",
                "str",
                ",",
                " ",
                "y",
                ":",
                " ",
                "&",
                "'b",
                " ",
                "str",
            ],
            tk,
        )
        self.assertEqual("an", ta[4])
        self.assertEqual("an", ta[13])

    def test_php_assoc(self):
        t = "'attr' => 'xyz1234?'"
        tk, ta = process_regex(t)
        self.assertListEqual(["'attr'", " ", "=>", " ", "'xyz1234?'"], tk)
        self.assertEqual("st", ta[0])
        self.assertEqual("st", ta[4])

    def test_empty_singlestr(self):
        t = "x = ''"
        tk, ta = process_regex(t)
        self.assertListEqual(["x", " ", "=", " ", "''"], tk)
        self.assertEqual("st", ta[4])

    def test_empty_doublestr(self):
        t = 'x = ""'
        tk, ta = process_regex(t)
        self.assertListEqual(["x", " ", "=", " ", '""'], tk)
        self.assertEqual("st", ta[4])

    def test_empty_triplesstr(self):
        t = '""""""'
        tk, ta = process_regex(t)
        self.assertListEqual(['""""""'], tk)
        self.assertEqual("st", ta[0])

    def test_py_docstr(self):
        t = 'def y(x):\n    """This is a..."""\n'
        tk, ta = process_regex(t)
        self.assertListEqual(
            [
                "def",
                " ",
                "y",
                "(",
                "x",
                ")",
                ":",
                "\n",
                "    ",
                '"""This is a..."""',
                "\n",
            ],
            tk,
        )

    def test_py_docstr_ml(self):
        t = 'def y():\n    """This is a...\n    ## parameters\n"""'
        tk, ta = process_regex(t)
        self.assertListEqual(
            [
                "def",
                " ",
                "y",
                "(",
                ")",
                ":",
                "\n",
                "    ",
                '"""This is a...\n    ## parameters\n"""',
            ],
            tk,
        )

    def test_singlestring(self):
        tk, ta = process_regex("s = [ 'some ']")
        self.assertEqual(["s", " ", "=", " ", "[", " ", "'some '", "]"], tk)
        self.assertEqual("st", ta[6])

    def test_strs_weird(self):
        tk, ta = process_regex("&x<> = 'some' , 'b '")
        self.assertListEqual(
            ["&", "x", "<", ">", " ", "=", " ", "'some'", " ", ",", " ", "'b '"], tk
        )
        self.assertEqual("st", ta[7])
        self.assertEqual("st", ta[11])

    def test_bash_vars(self):
        tk, _ = process_regex("$1 = 'a'")
        self.assertListEqual(["$1", " ", "=", " ", "'a'"], tk)
        tk, _ = process_regex("$? = 'a'")
        self.assertListEqual(["$?", " ", "=", " ", "'a'"], tk)

    def test_bash_specials(self):
        tk, _ = process_regex("$? $* $# $@ $- $$")
        self.assertListEqual(
            ["$?", " ", "$*", " ", "$#", " ", "$@", " ", "$-", " ", "$$"], tk
        )

    def test_bash_flag(self):
        tk, _ = process_regex("mkdir -p")
        self.assertListEqual(["mkdir", " ", "-p"], tk)

    def test_bash_flag_w_val(self):
        tk, _ = process_regex("--color=auto")
        self.assertListEqual(["--color", "=", "auto"], tk)

    def test_py_annot(self):
        tk, ta = process_regex("@staticmethod\ndef load_project(name)")
        self.assertListEqual(
            ["@staticmethod", "\n", "def", " ", "load_project", "(", "name", ")"], tk
        )

    def test_weirdquote(self):
        tk, ta = process_regex("=“OK”")
        self.assertListEqual(["=", "“OK”"], tk)

    def test_cssclass(self):
        tk, ta = process_regex(".my-class {\n")
        self.assertListEqual([".my-class", " ", "{", "\n"], tk)

    def test_css_attr(self):
        tk, ta = process_regex("  background-color: red")
        self.assertListEqual(["  ", "background-color", ":", " ", "red"], tk)

    def test_htmlco(self):
        tk, ta = process_regex("<!-- hej -->")
        self.assertListEqual(["<!-- hej -->"], tk)
        self.assertListEqual(["co"], ta)

    def test_co_in_fn(self):
        tk, ta = process_regex("{\n  a.b(); // OK\n}")
        self.assertListEqual("{|\n|  |a|.|b|(|)|;| |// OK|\n|}".split("|"), tk)
        self.assertEqual("brcl", ta[-1])
        self.assertEqual("nl", ta[-2])
        self.assertEqual("co", ta[-3])

    def test_docstr_end(self):
        tk, ta = process_regex('"""A function."""\n    variable')
        self.assertListEqual('"""A function."""|\n|    |variable'.split("|"), tk)
        self.assertEqual("st", ta[0])

    def test_modop_nospace(self):
        tk, ta = process_regex("8%2")
        self.assertListEqual(["8", "%", "2"], tk)


class TestMergeAdjacent(unittest.TestCase):
    def test_nomerge(self):
        tk = ["a", "b", "b"]
        ta = ["x", "y", "z"]
        tkm, tam, midx = text_process.merge_adjacent(tk, ta)
        self.assertListEqual(tk, tkm)
        self.assertListEqual(ta, tam)
        self.assertListEqual([], midx)

    def test_merge1(self):
        tk = ["a", "b", "b"]
        ta = ["x", "x", "y"]
        tkm, tam, midx = text_process.merge_adjacent(tk, ta)
        self.assertListEqual(["ab", "b"], tkm)
        self.assertListEqual(["x", "y"], tam)
        self.assertListEqual([0], midx)

    def test_merge2(self):
        tk = ["a", "b", "c", "d", "e", "f"]
        ta = ["x", "x", "x", "yes", "yes", "x"]
        tkm, tam, midx = text_process.merge_adjacent(tk, ta)
        self.assertListEqual(["abc", "de", "f"], tkm)
        self.assertListEqual(["x", "yes", "x"], tam)
        self.assertListEqual([0, 1], midx)

    def test_merge_excl(self):
        tk = ["a", "b", "c", "d"]
        ta = ["y", "y", "x", "x"]
        tkm, tam, midx = text_process.merge_adjacent(tk, ta, merge_only=["x", "z"])
        self.assertListEqual(["a", "b", "cd"], tkm)
        self.assertListEqual(["y", "y", "x"], tam)
        self.assertListEqual([2], midx)


class TestInferIndent(unittest.TestCase):
    def test_none(self):
        self.assertEqual(None, text_process.infer_indent("print(x)\na=b+c"))

    def test_4space(self):
        self.assertEqual(" " * 4, text_process.infer_indent("    print(x)"))
        self.assertEqual(" " * 4, text_process.infer_indent(" " * 8 + "print(x)"))

    def test_2_space(self):
        self.assertEqual(" " * 2, text_process.infer_indent("  a = b"))

    def test_2_4_space(self):
        self.assertEqual(
            " " * 2,
            text_process.infer_indent("a = b\n  for x:\n    print(x)"),
        )

    def test_1_2_3_space(self):
        self.assertEqual(" ", text_process.infer_indent(" hej\n  abc\n   xyz"))

    def test_tab(self):
        self.assertEqual("\t", text_process.infer_indent("\t123"), "single tab")
        self.assertEqual("\t", text_process.infer_indent("\t\t123"), "double tab")

    def test_php_coml(self):
        t = "/**\n * Get thse thing\n */\n$x = 1"
        self.assertEqual(None, text_process.infer_indent(t))


class TestBracLevel(unittest.TestCase):
    def test_none(self):
        t = ["va", "opas", "nu", "pu"]
        tn, levels = text_process.bracket_levels(t)
        self.assertListEqual(t, tn)
        self.assertListEqual([0, 0, 0, 0], levels)

    def test_1(self):
        t = ["fnfr", "brop", "va", "brcl"]
        tn, levels = text_process.bracket_levels(t)
        self.assertListEqual(["fnfr", "br0", "va", "br0"], tn)
        self.assertListEqual([0, 0, 1, 0], levels)

    def test_2(self):
        t = ["fnfr", "brop", "va", "opbi", "va", "brop", "nu", "brcl", "brcl"]
        tn, levels = text_process.bracket_levels(t)
        self.assertListEqual(
            ["fnfr", "br0", "va", "opbi", "va", "br1", "nu", "br1", "br0"], tn
        )
        self.assertListEqual([0, 0, 1, 1, 1, 1, 2, 1, 0], levels)


if __name__ == "__main__":
    unittest.main()
