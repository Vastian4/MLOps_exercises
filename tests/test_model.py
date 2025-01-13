import torch
import pytest

from src.code_structures_project.model import my_model

@pytest.mark.parametrize("batch_size", [16, 32, 64, 128])
def test_model(batch_size):
    model = my_model()
    x = torch.randn(batch_size, 1, 28, 28)
    y = model(x)
    assert y.shape == (batch_size, 10)

def test_raises():
    with pytest.raises(ValueError):
        model = my_model()
        x = torch.randn(1, 2, 3)
        output = model.forward(x)

