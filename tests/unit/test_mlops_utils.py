from pathlib import Path

from omegaconf import OmegaConf

from ncf_recommender.data.datasets import dataset_fingerprint
from ncf_recommender.training.trainer import _resolve_resume_checkpoint


def _write_file(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_dataset_fingerprint_stable(tmp_path: Path) -> None:
    base = tmp_path / "toy"
    _write_file(base.with_suffix(".train.rating"), "0\t1\t1\t0\n")
    _write_file(base.with_suffix(".test.rating"), "0\t2\t1\t0\n")
    _write_file(base.with_suffix(".test.negative"), "(0,2)\t3\t4\t5\n")
    f1 = dataset_fingerprint(tmp_path, "toy")
    f2 = dataset_fingerprint(tmp_path, "toy")
    assert f1["sha256"] == f2["sha256"]


def test_resolve_resume_checkpoint_default(tmp_path: Path) -> None:
    cfg = OmegaConf.create({"resume": {"checkpoint_path": ""}})
    default_last = tmp_path / "last.pt"
    assert _resolve_resume_checkpoint(cfg, default_last) == default_last


def test_resolve_resume_checkpoint_explicit(tmp_path: Path) -> None:
    explicit = tmp_path / "resume.pt"
    cfg = OmegaConf.create({"resume": {"checkpoint_path": str(explicit)}})
    default_last = tmp_path / "last.pt"
    assert _resolve_resume_checkpoint(cfg, default_last) == explicit
