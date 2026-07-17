from html_highlight.data_augmentation import modify_name, randomize_names


class TestModName:
    def test_var_php(self):
        x = modify_name("$xyz")
        assert x[0] == "$"
        assert x[1:].islower()

    def test_camel(self):
        x = modify_name("AbraCadabra")
        assert x[0].isupper()
        assert x[1].islower()
        assert x[4].isupper()
        assert x[5].islower()

    def test_snake(self):
        x = modify_name("a_snake")
        assert x.islower()
        assert x[1] == "_"


class TestRandomizeTokens:
    def test_fncall(self):
        x = randomize_names(
            ["x", "=", "f_un", "(", "x", ")"],
            ["va", "opas", "fnsa", "brop", "va", "brcl"],
        )
        assert x[1] == "="
        assert x[3] == "("
        assert x[5] == ")"
