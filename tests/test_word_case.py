from pytest import mark

from src.text_process import WordCase, get_word_case


@mark.parametrize("w", ["MyClass", "Cool"])
def test_upper_camel(w):
    assert get_word_case(w) == WordCase.UPPER_CAMEL


@mark.parametrize("w", ["myFunc", "aBunchOfData"])
def test_lower_camel(w):
    assert get_word_case(w) == WordCase.LOWER_CAMEL


@mark.parametrize("w", ["my-component"])
def test_kebab(w):
    assert get_word_case(w) == WordCase.KEBAB
