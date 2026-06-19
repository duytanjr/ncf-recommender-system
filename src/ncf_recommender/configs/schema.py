"""Typed config schemas for the NCF project."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TrainConfig:
    """High-level training config."""

    seed: int = 42
    model: str = "gmf"
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-3
