import pytest

from ncf_recommender.training.trainer import resolve_device


def test_resolve_device_cpu() -> None:
    device = resolve_device("cpu")
    assert str(device) == "cpu"


def test_resolve_device_invalid() -> None:
    with pytest.raises(ValueError):
        resolve_device("tpu")
