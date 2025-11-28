"""Tests for computing overlap between examples and datasets."""

from src import data_functions as dfun


def test_2grams_simple():
    ngrams = dfun.get_ngrams(["hello", "world", "123", "cool"], 2)
    assert ngrams == {("hello", "world"), ("world", "123"), ("123", "cool")}


def test_3grams_simple():
    ngrams = dfun.get_ngrams(["hello", "world", "123", "cool"], 3)
    assert ngrams == {("hello", "world", "123"), ("world", "123", "cool")}


class TestGetOverlap:
    def test_equal_iou(self):
        ovr = dfun.get_overlap(set("abc"), set("abc"), norm="iou")
        assert ovr == 1.0

    def test_equal_max(self):
        ovr = dfun.get_overlap(set("abc"), set("abc"), norm="max")
        assert ovr == 1.0

    def test_zero_iou(self):
        ovr = dfun.get_overlap(set("abc"), set("def"), norm="iou")
        assert ovr == 0.0

    def test_zero_max(self):
        ovr = dfun.get_overlap(set("abc"), set("def"), norm="max")
        assert ovr == 0.0

    def test_tiny_iou(self):
        ovr = dfun.get_overlap(set("ab"), set("bc"), norm="iou")
        assert ovr == 1 / 3

    def test_tiny_max(self):
        ovr = dfun.get_overlap(set("ab"), set("bc"), norm="max")
        assert ovr == 0.5
