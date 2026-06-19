from pathlib import Path

import torch
from omegaconf import OmegaConf

from ncf_recommender.inference.service import recommend_top_k
from ncf_recommender.models.gmf import GMF


def _write_file(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_recommend_top_k_returns_items(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    base = data_dir / "toy"
    _write_file(base.with_suffix(".train.rating"), "0\t1\t1\t0\n0\t2\t1\t0\n1\t3\t1\t0\n")
    _write_file(base.with_suffix(".test.rating"), "0\t4\t1\t0\n1\t5\t1\t0\n")
    _write_file(base.with_suffix(".test.negative"), "(0,4)\t6\t7\t8\n(1,5)\t9\t10\t11\n")

    model = GMF(num_users=2, num_items=4, embedding_dim=4)
    ckpt = tmp_path / "toy_gmf_best.pt"
    torch.save(
        {
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "metrics": {},
            "config": {},
        },
        ckpt,
    )

    cfg = OmegaConf.create(
        {
            "data": {"root_dir": str(data_dir), "name": "toy"},
            "model": {"name": "gmf", "embedding_dim": 4},
            "train": {"output_dir": str(tmp_path)},
            "eval": {"checkpoint_path": str(ckpt)},
        }
    )
    recs = recommend_top_k(cfg=cfg, user_id=0, top_k=2)
    assert len(recs) == 2
    assert 1 not in recs
    assert 2 not in recs
