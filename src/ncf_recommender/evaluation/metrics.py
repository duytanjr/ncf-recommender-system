"""Ranking metrics for implicit recommendation."""

from __future__ import annotations

import math
from collections.abc import Sequence


def hit_rate_at_k(ranklist: Sequence[int], gt_item: int) -> float:
    """Hit rate at k for one user."""
    return 1.0 if gt_item in ranklist else 0.0


def ndcg_at_k(ranklist: Sequence[int], gt_item: int) -> float:
    """NDCG at k for one user."""
    for index, item in enumerate(ranklist):
        if item == gt_item:
            return math.log(2.0) / math.log(index + 2.0)
    return 0.0


def recall_at_k(ranklist: Sequence[int], ground_truth_items: set[int]) -> float:
    """Recall at k for one user."""
    if not ground_truth_items:
        return 0.0
    hits = sum(1 for item in ranklist if item in ground_truth_items)
    return float(hits) / float(len(ground_truth_items))


def mrr_at_k(ranklist: Sequence[int], gt_item: int) -> float:
    """Mean reciprocal rank at k for one user."""
    for index, item in enumerate(ranklist):
        if item == gt_item:
            return 1.0 / float(index + 1)
    return 0.0
