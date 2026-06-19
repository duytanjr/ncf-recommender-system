"""Micro-benchmark for forward pass throughput."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import torch
from omegaconf import DictConfig

from ncf_recommender.data.datasets import load_legacy_implicit_dataset
from ncf_recommender.training.trainer import _build_model, _resolve_checkpoint_path, resolve_device


def benchmark_model(
    cfg: DictConfig,
    batch_size: int = 4096,
    steps: int = 200,
    warmup_steps: int = 20,
) -> dict[str, float]:
    """Benchmark inference throughput."""
    device = resolve_device(str(cfg.runtime.device))
    dataset = load_legacy_implicit_dataset(root=Path(cfg.data.root_dir), name=cfg.data.name)
    model = _build_model(cfg, dataset).to(device)

    ckpt = _resolve_checkpoint_path(cfg)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    users = torch.randint(0, dataset.num_users, (batch_size,), device=device)
    items = torch.randint(0, dataset.num_items, (batch_size,), device=device)

    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = model(users, items)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        start = perf_counter()
        for _ in range(steps):
            _ = model(users, items)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        duration = perf_counter() - start

    total_samples = batch_size * steps
    throughput = total_samples / max(duration, 1e-6)
    latency_ms = (duration / steps) * 1000.0
    result = {
        "throughput_samples_per_s": throughput,
        "latency_ms_per_batch": latency_ms,
        "device": str(device),
    }
    return result
