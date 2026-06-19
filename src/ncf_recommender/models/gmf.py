"""Generalized Matrix Factorization model."""

from __future__ import annotations

import torch
from torch import nn


class GMF(nn.Module):
    """Minimal GMF skeleton."""

    def __init__(self, num_users: int, num_items: int, embedding_dim: int) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.output = nn.Linear(embedding_dim, 1)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        user_vec = self.user_embedding(user_ids)
        item_vec = self.item_embedding(item_ids)
        logits = self.output(user_vec * item_vec)
        return logits.squeeze(-1)
