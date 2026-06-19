from pathlib import Path

import torch
from omegaconf import OmegaConf

from ncf_recommender.models.gmf import GMF
from ncf_recommender.training.trainer import _resolve_checkpoint_path, _save_checkpoint


def test_checkpoint_save_and_load_roundtrip(tmp_path: Path) -> None:
    model = GMF(num_users=8, num_items=16, embedding_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    cfg = OmegaConf.create(
        {
            "project": {"seed": 42},
            "data": {"name": "toy"},
            "model": {"name": "gmf"},
            "train": {"output_dir": str(tmp_path)},
            "eval": {"checkpoint_path": ""},
        }
    )
    ckpt = tmp_path / "toy_gmf_best.pt"
    _save_checkpoint(
        path=ckpt,
        epoch=3,
        model=model,
        optimizer=optimizer,
        metrics={"ndcg@k": 0.5},
        cfg=cfg,
    )
    loaded = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert loaded["epoch"] == 3
    assert "model_state_dict" in loaded
    assert loaded["metrics"]["ndcg@k"] == 0.5


def test_resolve_checkpoint_prefers_eval_path(tmp_path: Path) -> None:
    explicit = tmp_path / "custom.pt"
    cfg = OmegaConf.create(
        {
            "data": {"name": "ml-1m"},
            "model": {"name": "gmf"},
            "train": {"output_dir": str(tmp_path / "artifacts")},
            "eval": {"checkpoint_path": str(explicit)},
        }
    )
    assert _resolve_checkpoint_path(cfg) == explicit


def test_resolve_checkpoint_uses_default_pattern(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "data": {"name": "ml-1m"},
            "model": {"name": "gmf"},
            "train": {"output_dir": str(tmp_path / "artifacts")},
            "eval": {"checkpoint_path": ""},
        }
    )
    expected = Path(str(cfg.train.output_dir)) / "ml-1m_gmf_best.pt"
    assert _resolve_checkpoint_path(cfg) == expected
