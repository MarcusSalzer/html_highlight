import sys

import pytest
import torch

sys.path.append("src")

from src.tensor_sequence_dataset import TensorSequenceDataset


def test_init():
    # Test case 1: Creating a dataset with some tokens
    tokens = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
    tags = [torch.tensor([7, 8, 9]), torch.tensor([10, 11])]
    tags_det = [torch.tensor([13, 14, 15]), torch.tensor([16, 17])]

    dataset = TensorSequenceDataset(tokens, tags, tags_det)

    assert len(dataset) == 2, "length"

    # Accessing an item in the dataset
    first_item = dataset[0]

    assert first_item["tokens"].equal(torch.tensor([1, 2, 3]))
    assert first_item["tags"].equal(torch.tensor([7, 8, 9]))
    assert first_item["tags_det"].equal(torch.tensor([13, 14, 15]))


def test_validation():
    # weird lengths
    with pytest.raises(AssertionError):
        TensorSequenceDataset(
            [torch.tensor([1, 2, 3])],
            [torch.tensor([7, 8, 9])],
            [torch.tensor([13, 14, 15]), torch.tensor([16, 17])],
        )
    # floats
    with pytest.raises(AssertionError):
        TensorSequenceDataset(
            [torch.tensor([1.5, 2, 3])],
            [torch.tensor([7, 8, 9])],
            [torch.tensor([13, 4.4, 15])],
        )
