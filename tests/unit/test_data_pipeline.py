from pathlib import Path

import pytest

from ncf_recommender.data.datasets import load_legacy_implicit_dataset


def _write_file(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_load_legacy_dataset_triplet(tmp_path: Path) -> None:
    base = tmp_path / "toy"
    _write_file(
        base.with_suffix(".train.rating"),
        "0\t1\t1\t0\n0\t2\t1\t0\n1\t3\t1\t0\n",
    )
    _write_file(
        base.with_suffix(".test.rating"),
        "0\t4\t1\t0\n1\t5\t1\t0\n",
    )
    _write_file(
        base.with_suffix(".test.negative"),
        "(0,4)\t6\t7\t8\n(1,5)\t9\t10\t11\n",
    )

    ds = load_legacy_implicit_dataset(root=tmp_path, name="toy")
    assert ds.num_users == 2
    assert ds.num_items == 4
    assert ds.train_matrix[0, 1] == 1.0
    assert ds.test_ratings == [(0, 4), (1, 5)]
    assert ds.test_negatives[0] == [6, 7, 8]


def test_load_legacy_dataset_mismatched_lengths(tmp_path: Path) -> None:
    base = tmp_path / "bad"
    _write_file(base.with_suffix(".train.rating"), "0\t1\t1\t0\n")
    _write_file(base.with_suffix(".test.rating"), "0\t2\t1\t0\n1\t3\t1\t0\n")
    _write_file(base.with_suffix(".test.negative"), "(0,2)\t4\t5\t6\n")

    with pytest.raises(ValueError):
        load_legacy_implicit_dataset(root=tmp_path, name="bad")
