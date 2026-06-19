"""Model export helpers (TorchScript / ONNX)."""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig

from ncf_recommender.data.datasets import load_legacy_implicit_dataset
from ncf_recommender.training.trainer import _build_model, _resolve_checkpoint_path, resolve_device


def export_model(
    cfg: DictConfig,
    output_path: Path,
    export_format: str = "torchscript",
    checkpoint_path: Path | None = None,
) -> Path:
    """Export a trained model to production artifact."""
    device = resolve_device(str(cfg.runtime.device))
    dataset = load_legacy_implicit_dataset(root=Path(cfg.data.root_dir), name=cfg.data.name)
    model = _build_model(cfg, dataset).to(device)

    ckpt = checkpoint_path or _resolve_checkpoint_path(cfg)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_user = torch.zeros((1,), dtype=torch.long, device=device)
    dummy_item = torch.zeros((1,), dtype=torch.long, device=device)

    fmt = export_format.strip().lower()
    if fmt == "torchscript":
        traced = torch.jit.trace(model, (dummy_user, dummy_item))
        traced.save(str(output_path))
        return output_path

    if fmt == "onnx":
        torch.onnx.export(
            model,
            (dummy_user, dummy_item),
            str(output_path),
            input_names=["user_ids", "item_ids"],
            output_names=["scores"],
            dynamic_axes={
                "user_ids": {0: "batch"},
                "item_ids": {0: "batch"},
                "scores": {0: "batch"},
            },
            opset_version=17,
        )
        return output_path

    raise ValueError(f"Unsupported export format: {export_format}")
