"""Pointwise negative sampling helper for training."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class PointwiseImplicitDataset(Dataset):
    """Custom PyTorch dataset wrapping pointwise interactions."""

    def __init__(self, user_ids: np.ndarray, item_ids: np.ndarray, labels: np.ndarray) -> None:
        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.item_ids = torch.tensor(item_ids, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.user_ids[idx], self.item_ids[idx], self.labels[idx]


def build_pointwise_samples(
    interactions: list[tuple[int, int]],
    observed: set[tuple[int, int]],
    num_items: int,
    num_negatives: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample negatives for each user positive interaction."""
    user_list = []
    item_list = []
    label_list = []

    for u, i in interactions:
        # Positive sample
        user_list.append(u)
        item_list.append(i)
        label_list.append(1.0)

        # Negative samples
        for _ in range(num_negatives):
            j = rng.integers(0, num_items)
            while (u, j) in observed:
                j = rng.integers(0, num_items)
            user_list.append(u)
            item_list.append(j)
            label_list.append(0.0)

    return (
        np.array(user_list, dtype=np.int64),
        np.array(item_list, dtype=np.int64),
        np.array(label_list, dtype=np.float32),
    )
