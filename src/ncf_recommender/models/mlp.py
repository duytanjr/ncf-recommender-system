"""MLP-based collaborative filtering model."""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """Minimal MLP skeleton."""

    def __init__(self, num_users: int, num_items: int, layer_sizes: list[int]) -> None:
        super().__init__()
        embed_dim = layer_sizes[0] // 2
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)

        mlp_layers: list[nn.Module] = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:], strict=False):
            mlp_layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
        self.mlp = nn.Sequential(*mlp_layers)
        self.output = nn.Linear(layer_sizes[-1], 1)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        user_vec = self.user_embedding(user_ids)
        item_vec = self.item_embedding(item_ids)
        x = torch.cat([user_vec, item_vec], dim=-1)
        x = self.mlp(x)
        logits = self.output(x)
        return logits.squeeze(-1)
