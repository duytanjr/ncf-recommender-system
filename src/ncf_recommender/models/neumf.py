"""Neural Matrix Factorization model."""

from __future__ import annotations

import torch
from torch import nn


class NeuMF(nn.Module):
    """Minimal NeuMF skeleton."""

    def __init__(self, num_users: int, num_items: int, mf_dim: int, mlp_layers: list[int]) -> None:
        super().__init__()
        mlp_embed_dim = mlp_layers[0] // 2

        self.mf_user_embedding = nn.Embedding(num_users, mf_dim)
        self.mf_item_embedding = nn.Embedding(num_items, mf_dim)
        self.mlp_user_embedding = nn.Embedding(num_users, mlp_embed_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, mlp_embed_dim)

        mlp_modules: list[nn.Module] = []
        for in_dim, out_dim in zip(mlp_layers[:-1], mlp_layers[1:], strict=False):
            mlp_modules.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
        self.mlp = nn.Sequential(*mlp_modules)

        self.output = nn.Linear(mf_dim + mlp_layers[-1], 1)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        mf_vector = self.mf_user_embedding(user_ids) * self.mf_item_embedding(item_ids)

        mlp_user = self.mlp_user_embedding(user_ids)
        mlp_item = self.mlp_item_embedding(item_ids)
        mlp_vector = self.mlp(torch.cat([mlp_user, mlp_item], dim=-1))

        logits = self.output(torch.cat([mf_vector, mlp_vector], dim=-1))
        return logits.squeeze(-1)
