import pytest

from ncf_recommender.training.trainer import _monitor_to_metric_key


def test_monitor_mapping() -> None:
    assert _monitor_to_metric_key("val_ndcg@10") == "ndcg@k"
    assert _monitor_to_metric_key("val_hr@10") == "hr@k"


def test_monitor_mapping_invalid() -> None:
    with pytest.raises(ValueError):
        _monitor_to_metric_key("val_auc")
