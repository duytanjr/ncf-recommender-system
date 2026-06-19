"""Inference helper APIs for top-k retrieval."""

from __future__ import annotations

import heapq
from pathlib import Path

import torch
from omegaconf import DictConfig

from ncf_recommender.data.datasets import load_legacy_implicit_dataset
from ncf_recommender.training.trainer import _build_model, _resolve_checkpoint_path, resolve_device


@torch.no_grad()
def recommend_top_k(
    cfg: DictConfig,
    user_id: int,
    top_k: int,
    checkpoint_path: Path | None = None,
) -> list[int]:
    """Recommend top-k items for one user using trained checkpoint."""
    recs = recommend_top_k_with_scores(
        cfg=cfg,
        user_id=user_id,
        top_k=top_k,
        checkpoint_path=checkpoint_path,
    )
    return [item_id for item_id, _ in recs]


@torch.no_grad()
def recommend_top_k_with_scores(
    cfg: DictConfig,
    user_id: int,
    top_k: int,
    checkpoint_path: Path | None = None,
) -> list[tuple[int, float]]:
    """Recommend top-k item ids with ranking scores for one user."""
    runtime_cfg = cfg.get("runtime")
    device_policy = "auto"
    if runtime_cfg is not None and "device" in runtime_cfg:
        device_policy = str(runtime_cfg.device)
    device = resolve_device(device_policy)
    dataset = load_legacy_implicit_dataset(
        root=Path(cfg.data.root_dir),
        name=cfg.data.name,
    )
    if user_id < 0 or user_id >= dataset.num_users:
        raise ValueError(f"user_id={user_id} out of range [0, {dataset.num_users - 1}]")

    model = _build_model(cfg, dataset).to(device)
    ckpt = checkpoint_path or _resolve_checkpoint_path(cfg)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    interacted_items = {item for (user, item) in dataset.train_matrix.keys() if user == user_id}
    candidate_items = [item for item in range(dataset.num_items) if item not in interacted_items]
    if not candidate_items:
        return []

    users = torch.full((len(candidate_items),), fill_value=user_id, dtype=torch.long, device=device)
    items = torch.tensor(candidate_items, dtype=torch.long, device=device)
    logits = model(users, items)
    scores = torch.sigmoid(logits).detach().cpu().tolist()

    item_to_score = dict(zip(candidate_items, scores, strict=False))
    top_items = heapq.nlargest(top_k, item_to_score, key=item_to_score.get)
    return [(item_id, float(item_to_score[item_id])) for item_id in top_items]


def get_user_interaction_history(cfg: DictConfig, user_id: int) -> list[int]:
    """Return interacted item ids from train matrix for one user."""
    dataset = load_legacy_implicit_dataset(
        root=Path(cfg.data.root_dir),
        name=cfg.data.name,
    )
    if user_id < 0 or user_id >= dataset.num_users:
        raise ValueError(f"user_id={user_id} out of range [0, {dataset.num_users - 1}]")
    return sorted(item for (user, item) in dataset.train_matrix.keys() if user == user_id)


def get_dataset_stats(cfg: DictConfig) -> dict[str, int]:
    """Return basic dataset stats for UI display."""
    dataset = load_legacy_implicit_dataset(
        root=Path(cfg.data.root_dir),
        name=cfg.data.name,
    )
    return {
        "num_users": dataset.num_users,
        "num_items": dataset.num_items,
        "num_train_interactions": int(dataset.train_matrix.nnz),
        "num_test_instances": len(dataset.test_ratings),
    }
