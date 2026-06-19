"""Dataset definitions and legacy loaders."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.sparse import dok_matrix


class ImplicitDataset:
    """Container for legacy implicit rating split files."""

    def __init__(
        self,
        train_matrix: dok_matrix,
        test_ratings: list[tuple[int, int]],
        test_negatives: list[list[int]],
        num_users: int,
        num_items: int,
    ) -> None:
        self.train_matrix = train_matrix
        self.test_ratings = test_ratings
        self.test_negatives = test_negatives
        self.num_users = num_users
        self.num_items = num_items


def load_legacy_implicit_dataset(root: Path, name: str) -> ImplicitDataset:
    """Load train.rating, test.rating and test.negative splits."""
    train_path = root / f"{name}.train.rating"
    test_path = root / f"{name}.test.rating"
    neg_path = root / f"{name}.test.negative"

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")
    if not neg_path.exists():
        raise FileNotFoundError(f"Negative file not found: {neg_path}")

    # Parse train ratings first to get dimensions and build matrix
    train_pairs: list[tuple[int, int]] = []
    max_user = -1
    max_item = -1
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                u = int(parts[0])
                i = int(parts[1])
                train_pairs.append((u, i))
                if u > max_user:
                    max_user = u
                if i > max_item:
                    max_item = i

    num_users = max_user + 1
    num_items = max_item + 1

    train_matrix = dok_matrix((num_users, num_items), dtype=np.float32)
    for u, i in train_pairs:
        train_matrix[u, i] = 1.0

    # Parse test ratings
    test_ratings: list[tuple[int, int]] = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                u = int(parts[0])
                i = int(parts[1])
                test_ratings.append((u, i))

    # Parse test negatives
    test_negatives: list[list[int]] = []
    with open(neg_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 1:
                # The first part is (user,gt_item)
                neg_items = [int(x) for x in parts[1:]]
                test_negatives.append(neg_items)

    if len(test_ratings) != len(test_negatives):
        raise ValueError(
            f"Mismatched lengths: test_ratings has {len(test_ratings)} entries, "
            f"test_negatives has {len(test_negatives)} entries."
        )

    return ImplicitDataset(
        train_matrix=train_matrix,
        test_ratings=test_ratings,
        test_negatives=test_negatives,
        num_users=num_users,
        num_items=num_items,
    )


def dataset_fingerprint(root: Path, name: str) -> dict[str, str]:
    """Compute sha256 checksum across dataset files."""
    sha256 = hashlib.sha256()
    for suffix in [".train.rating", ".test.rating", ".test.negative"]:
        path = root / f"{name}{suffix}"
        if path.exists():
            with open(path, "rb") as f:
                while chunk := f.read(8192):
                    sha256.update(chunk)
    return {"sha256": sha256.hexdigest()}
