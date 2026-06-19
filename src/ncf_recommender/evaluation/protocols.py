"""Evaluation protocols (leave-one-out and beyond)."""

from __future__ import annotations

import heapq
from collections.abc import Mapping

import numpy as np
import torch
from torch import nn

from ncf_recommender.evaluation.metrics import hit_rate_at_k, mrr_at_k, ndcg_at_k, recall_at_k


@torch.no_grad()
def evaluate_leave_one_out(
    model: nn.Module,
    test_ratings: list[tuple[int, int]],
    test_negatives: list[list[int]],
    top_k: int,
    device: torch.device,
    batch_size: int = 1024,
) -> dict[str, float]:
    """Evaluate legacy leave-one-out protocol."""
    model.eval()
    hrs: list[float] = []
    ndcgs: list[float] = []
    recalls: list[float] = []
    mrrs: list[float] = []

    for rating, negatives in zip(test_ratings, test_negatives, strict=False):
        user_id, gt_item = rating
        candidate_items = list(negatives)
        candidate_items.append(gt_item)

        users = torch.full(
            size=(len(candidate_items),),
            fill_value=user_id,
            dtype=torch.long,
            device=device,
        )
        items = torch.tensor(candidate_items, dtype=torch.long, device=device)

        predictions: list[float] = []
        for start in range(0, len(candidate_items), batch_size):
            end = start + batch_size
            scores = model(users[start:end], items[start:end])
            predictions.extend(scores.detach().cpu().tolist())

        item_to_score: Mapping[int, float] = dict(zip(candidate_items, predictions, strict=False))
        ranklist = heapq.nlargest(top_k, item_to_score, key=item_to_score.get)

        hrs.append(hit_rate_at_k(ranklist, gt_item))
        ndcgs.append(ndcg_at_k(ranklist, gt_item))
        recalls.append(recall_at_k(ranklist, {gt_item}))
        mrrs.append(mrr_at_k(ranklist, gt_item))

    return {
        "hr@k": float(np.mean(hrs)),
        "ndcg@k": float(np.mean(ndcgs)),
        "recall@k": float(np.mean(recalls)),
        "mrr@k": float(np.mean(mrrs)),
    }
