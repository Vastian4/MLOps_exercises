from torch.utils.data import Dataset
import pytest

from code_structures_project.data import MyDataset

import os.path
@pytest.mark.skipif(not os.path.exists("data/raw"), reason="Data files not found")

def test_my_dataset():
    """Test the MyDataset class."""
    dataset = MyDataset("data/raw")
    assert isinstance(dataset, Dataset)
    assert dataset.len(true_if_train_false_if_test=True) == 30000, "Dataset does not have the right number of samples"
    assert dataset.len(true_if_train_false_if_test=False) == 5000, "Dataset does not have the right number of samples"

    for i in range(dataset.len(true_if_train_false_if_test=True)):
        assert dataset.train_data[i].shape == (1, 28, 28)
        assert dataset.train_labels[i] in range(10)

    for i in range(dataset.len(true_if_train_false_if_test=False)):
        assert dataset.test_data[i].shape == (1, 28, 28)
        assert dataset.test_labels[i] in range(10)
