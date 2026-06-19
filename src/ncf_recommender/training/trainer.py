"""Production training and evaluation orchestration."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from ncf_recommender.data.datasets import (
    ImplicitDataset,
    dataset_fingerprint,
    load_legacy_implicit_dataset,
)
from ncf_recommender.data.negative_sampling import PointwiseImplicitDataset, build_pointwise_samples
from ncf_recommender.evaluation.protocols import evaluate_leave_one_out
from ncf_recommender.models.gmf import GMF
from ncf_recommender.models.mlp import MLP
from ncf_recommender.models.neumf import NeuMF
from ncf_recommender.training.tracking import MlflowTracker, utc_now_iso, write_run_manifest


@dataclass(slots=True)
class TrainArtifacts:
    """Result bundle after training."""

    best_checkpoint: Path
    last_checkpoint: Path
    best_epoch: int
    best_metrics: dict[str, float]
    run_manifest: Path | None


def run_training(cfg: DictConfig) -> TrainArtifacts:
    """Run end-to-end training for GMF/MLP/NeuMF models."""
    seed = int(cfg.project.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    device = resolve_device(str(cfg.runtime.device))
    print(f"Runtime device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")

    dataset_root = Path(cfg.data.root_dir)
    dataset = load_legacy_implicit_dataset(
        root=dataset_root,
        name=cfg.data.name,
    )
    model = _build_model(cfg, dataset).to(device)

    if bool(cfg.runtime.compile_model) and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    if str(cfg.model.name).lower() == "neumf" and bool(cfg.model.use_pretrain):
        _warm_start_neumf_from_checkpoints(model=model, cfg=cfg, dataset=dataset, device=device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=float(cfg.train.learning_rate))

    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"{cfg.data.name}_{cfg.model.name}_best.pt"
    last_ckpt_path = output_dir / f"{cfg.data.name}_{cfg.model.name}_last.pt"

    start_epoch = 0
    if bool(cfg.resume.enabled):
        resume_path = _resolve_resume_checkpoint(cfg=cfg, default_last=last_ckpt_path)
        if resume_path.exists():
            start_epoch = _load_training_checkpoint(model, optimizer, resume_path, device) + 1
            print(f"Resumed from checkpoint: {resume_path} at epoch={start_epoch}")
        else:
            print(f"Resume enabled but checkpoint not found: {resume_path}")

    monitor_metric = _monitor_to_metric_key(str(cfg.train.early_stopping.monitor))
    monitor_mode = str(cfg.train.early_stopping.mode).lower()
    if monitor_mode not in {"min", "max"}:
        raise ValueError("train.early_stopping.mode must be one of: min, max")
    patience = int(cfg.train.early_stopping.patience)
    early_stopping_enabled = bool(cfg.train.early_stopping.enabled)
    early_stop_counter = 0
    writers = _init_writers(cfg)
    mlflow_tracker = MlflowTracker(
        enabled=bool(cfg.tracking.mlflow.enabled),
        tracking_uri=str(cfg.tracking.mlflow.tracking_uri),
        experiment_name=str(cfg.tracking.mlflow.experiment_name),
    )

    run_id = f"{cfg.model.name}_{cfg.data.name}_{uuid.uuid4().hex[:8]}"
    run_manifest: Path | None = None
    dataset_meta = (
        dataset_fingerprint(dataset_root, cfg.data.name)
        if bool(cfg.tracking.save_dataset_fingerprint)
        else {}
    )

    mlflow_tracker.log_params(
        {
            "model": str(cfg.model.name),
            "dataset": str(cfg.data.name),
            "device": str(device),
            "precision": str(cfg.runtime.precision),
            "seed": seed,
        }
    )

    interactions = list(dataset.train_matrix.keys())
    observed = set(interactions)

    best_epoch = -1
    best_metrics: dict[str, float] = {}
    best_monitor_value = float("-inf") if monitor_mode == "max" else float("inf")
    prof_ctx = _maybe_profiler(cfg, device)
    precision = str(cfg.runtime.precision).lower()
    amp_enabled = device.type == "cuda" and precision in {"fp16", "bf16"}
    scaler_enabled = device.type == "cuda" and precision == "fp16"
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    autocast_dtype = torch.float16 if precision == "fp16" else torch.bfloat16

    for epoch in range(start_epoch, int(cfg.train.epochs)):
        model.train()
        user_ids, item_ids, labels = build_pointwise_samples(
            interactions=interactions,
            observed=observed,
            num_items=dataset.num_items,
            num_negatives=int(cfg.train.num_negatives),
            rng=rng,
        )
        train_dataset = PointwiseImplicitDataset(user_ids, item_ids, labels)
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(cfg.train.batch_size),
            shuffle=True,
            num_workers=int(cfg.train.num_workers),
        )

        epoch_loss = 0.0
        sample_count = 0
        train_start = perf_counter()
        for step, (user_batch, item_batch, label_batch) in enumerate(
            tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{int(cfg.train.epochs)}",
                leave=False,
            )
        ):
            user_batch = user_batch.to(device)
            item_batch = item_batch.to(device)
            label_batch = label_batch.to(device)
            batch_size = int(label_batch.shape[0])

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                enabled=amp_enabled,
                dtype=autocast_dtype,
            ):
                preds = model(user_batch, item_batch)
                loss = criterion(preds, label_batch)
            if scaler_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += float(loss.item())
            sample_count += batch_size
            if prof_ctx is not None and step < int(cfg.profiling.max_profile_steps):
                prof_ctx.step()

        train_duration = perf_counter() - train_start
        throughput = float(sample_count) / max(train_duration, 1e-6)

        eval_metrics = evaluate_leave_one_out(
            model=model,
            test_ratings=dataset.test_ratings,
            test_negatives=dataset.test_negatives,
            top_k=int(cfg.eval.top_k),
            batch_size=int(cfg.eval.eval_batch_size),
            device=device,
        )
        avg_loss = epoch_loss / max(len(train_loader), 1)
        log_line = (
            f"epoch={epoch} loss={avg_loss:.5f} throughput={throughput:.1f} samples/s "
            f"HR@{int(cfg.eval.top_k)}={eval_metrics['hr@k']:.4f} "
            f"NDCG@{int(cfg.eval.top_k)}={eval_metrics['ndcg@k']:.4f} "
            f"MRR@{int(cfg.eval.top_k)}={eval_metrics['mrr@k']:.4f}"
        )
        print(log_line)
        logged_metrics = {
            "train/loss": avg_loss,
            "train/throughput_samples_per_s": throughput,
            "val/hr_at_k": eval_metrics["hr@k"],
            "val/ndcg_at_k": eval_metrics["ndcg@k"],
            "val/recall_at_k": eval_metrics["recall@k"],
            "val/mrr_at_k": eval_metrics["mrr@k"],
        }
        if device.type == "cuda":
            max_cuda_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
            logged_metrics["system/max_cuda_mem_mb"] = max_cuda_mb
        _log_metrics(
            writers=writers,
            metrics=logged_metrics,
            step=epoch,
        )
        mlflow_tracker.log_metrics(
            {
                "loss": avg_loss,
                "throughput_samples_per_s": throughput,
                "hr_at_k": eval_metrics["hr@k"],
                "ndcg_at_k": eval_metrics["ndcg@k"],
                "recall_at_k": eval_metrics["recall@k"],
                "mrr_at_k": eval_metrics["mrr@k"],
            },
            step=epoch,
        )

        _save_checkpoint(
            path=last_ckpt_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            metrics=eval_metrics,
            cfg=cfg,
        )

        current_value = float(eval_metrics[monitor_metric])
        if monitor_mode == "max":
            improved = current_value > best_monitor_value
        else:
            improved = current_value < best_monitor_value
        
        if improved:
            best_metrics = eval_metrics
            best_epoch = epoch
            best_monitor_value = current_value
            early_stop_counter = 0
            _save_checkpoint(
                path=ckpt_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                metrics=eval_metrics,
                cfg=cfg,
            )
        else:
            early_stop_counter += 1

        if early_stopping_enabled and early_stop_counter >= patience:
            print(f"Early stopping at epoch={epoch} with best_epoch={best_epoch}")
            break

    if prof_ctx is not None:
        prof_ctx.stop()
    _close_writers(writers)

    if bool(cfg.tracking.save_run_manifest):
        run_payload: dict[str, Any] = {
            "run_id": run_id,
            "finished_at_utc": utc_now_iso(),
            "model": str(cfg.model.name),
            "dataset": str(cfg.data.name),
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "best_checkpoint": str(ckpt_path),
            "last_checkpoint": str(last_ckpt_path),
            "device": str(device),
            "precision": str(cfg.runtime.precision),
            "dataset_fingerprint": dataset_meta,
            "config": _to_plain_dict(cfg),
        }
        run_manifest = write_run_manifest(Path(cfg.tracking.run_dir), run_payload)
        mlflow_tracker.log_artifact(run_manifest)
    mlflow_tracker.close()

    return TrainArtifacts(
        best_checkpoint=ckpt_path,
        last_checkpoint=last_ckpt_path,
        best_epoch=best_epoch,
        best_metrics=best_metrics,
        run_manifest=run_manifest,
    )


def run_evaluation(cfg: DictConfig, checkpoint_path: Path | None = None) -> dict[str, float]:
    """Evaluate a trained model checkpoint with legacy protocol."""
    device = resolve_device(str(cfg.runtime.device))
    dataset = load_legacy_implicit_dataset(
        root=Path(cfg.data.root_dir),
        name=cfg.data.name,
    )
    model = _build_model(cfg, dataset).to(device)

    ckpt = checkpoint_path or _resolve_checkpoint_path(cfg)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])

    return evaluate_leave_one_out(
        model=model,
        test_ratings=dataset.test_ratings,
        test_negatives=dataset.test_negatives,
        top_k=int(cfg.eval.top_k),
        batch_size=int(cfg.eval.eval_batch_size),
        device=device,
    )


def resolve_device(device_policy: str) -> torch.device:
    """Resolve runtime device from policy."""
    policy = device_policy.strip().lower()
    if policy == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if policy == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("runtime.device=cuda requested but CUDA is not available.")
        return torch.device("cuda")
    if policy == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported runtime.device policy: {device_policy}")


def _build_model(cfg: DictConfig, dataset: ImplicitDataset) -> nn.Module:
    model_name = str(cfg.model.name).lower()
    if model_name == "gmf":
        return GMF(
            num_users=dataset.num_users,
            num_items=dataset.num_items,
            embedding_dim=int(cfg.model.embedding_dim),
        )
    if model_name == "mlp":
        return MLP(
            num_users=dataset.num_users,
            num_items=dataset.num_items,
            layer_sizes=[int(value) for value in cfg.model.layers],
        )
    if model_name == "neumf":
        return NeuMF(
            num_users=dataset.num_users,
            num_items=dataset.num_items,
            mf_dim=int(cfg.model.mf_dim),
            mlp_layers=[int(value) for value in cfg.model.mlp_layers],
        )
    raise ValueError(f"Unsupported model name: {cfg.model.name}")


def _to_plain_dict(cfg: DictConfig) -> dict[str, Any]:
    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):
        raise TypeError("Expected DictConfig to convert into dict.")
    return container


def _save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: dict[str, float],
    cfg: DictConfig,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": _to_plain_dict(cfg),
        },
        path,
    )


def _load_training_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: Path,
    device: torch.device,
) -> int:
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    if "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return int(state.get("epoch", 0))


def _monitor_to_metric_key(monitor_name: str) -> str:
    normalized = monitor_name.strip().lower()
    mapping = {
        "val_ndcg@10": "ndcg@k",
        "val_hr@10": "hr@k",
        "val_recall@10": "recall@k",
        "val_mrr@10": "mrr@k",
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported monitor: {monitor_name}")
    return mapping[normalized]


def _init_writers(cfg: DictConfig) -> dict[str, Any]:
    writers: dict[str, Any] = {}
    logger_cfg = cfg.get("logging")
    if logger_cfg is None:
        return writers

    if bool(logger_cfg.tensorboard.enabled):
        try:
            from torch.utils.tensorboard import SummaryWriter

            log_dir = Path(str(logger_cfg.tensorboard.log_dir))
            log_dir.mkdir(parents=True, exist_ok=True)
            writers["tensorboard"] = SummaryWriter(log_dir=str(log_dir))
        except ImportError:
            print("TensorBoard logger requested but tensorboard is not installed.")

    if bool(logger_cfg.wandb.enabled):
        try:
            import wandb

            writers["wandb"] = wandb
            wandb.init(
                project=str(logger_cfg.wandb.project),
                name=str(logger_cfg.wandb.run_name),
                config=_to_plain_dict(cfg),
            )
        except ImportError:
            print("W&B logger requested but wandb is not installed.")

    return writers


def _log_metrics(writers: dict[str, Any], metrics: dict[str, float], step: int) -> None:
    tensorboard_writer = writers.get("tensorboard")
    if tensorboard_writer is not None:
        for key, value in metrics.items():
            tensorboard_writer.add_scalar(key, value, step)

    wandb_module = writers.get("wandb")
    if wandb_module is not None:
        wandb_module.log(metrics, step=step)


def _close_writers(writers: dict[str, Any]) -> None:
    tensorboard_writer = writers.get("tensorboard")
    if tensorboard_writer is not None:
        tensorboard_writer.close()

    wandb_module = writers.get("wandb")
    if wandb_module is not None:
        wandb_module.finish()


def _resolve_checkpoint_path(cfg: DictConfig) -> Path:
    checkpoint_path = str(cfg.eval.checkpoint_path)
    if checkpoint_path.strip():
        return Path(checkpoint_path)
    return Path(cfg.train.output_dir) / f"{cfg.data.name}_{cfg.model.name}_best.pt"


def _resolve_resume_checkpoint(cfg: DictConfig, default_last: Path) -> Path:
    checkpoint_path = str(cfg.resume.checkpoint_path)
    if checkpoint_path.strip():
        return Path(checkpoint_path)
    return default_last


def _maybe_profiler(cfg: DictConfig, device: torch.device) -> object | None:
    if not bool(cfg.profiling.enabled):
        return None
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    output_dir = Path(cfg.profiling.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_handler = torch.profiler.tensorboard_trace_handler(str(output_dir))
    prof = torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )
    prof.start()
    return prof


def _warm_start_neumf_from_checkpoints(
    model: nn.Module,
    cfg: DictConfig,
    dataset: ImplicitDataset,
    device: torch.device,
) -> None:
    if not isinstance(model, NeuMF):
        raise TypeError("Warm-start requires NeuMF model instance.")

    gmf_ckpt = Path(str(cfg.model.pretrain.gmf_checkpoint))
    mlp_ckpt = Path(str(cfg.model.pretrain.mlp_checkpoint))
    if not gmf_ckpt.exists() or not mlp_ckpt.exists():
        raise FileNotFoundError(
            "NeuMF pretrain requested but GMF/MLP checkpoint is missing. "
            f"gmf={gmf_ckpt}, mlp={mlp_ckpt}",
        )

    gmf_model = GMF(
        num_users=dataset.num_users,
        num_items=dataset.num_items,
        embedding_dim=int(cfg.model.mf_dim),
    ).to(device)
    mlp_model = MLP(
        num_users=dataset.num_users,
        num_items=dataset.num_items,
        layer_sizes=[int(value) for value in cfg.model.mlp_layers],
    ).to(device)

    gmf_state = torch.load(gmf_ckpt, map_location=device, weights_only=False)
    mlp_state = torch.load(mlp_ckpt, map_location=device, weights_only=False)
    gmf_model.load_state_dict(gmf_state["model_state_dict"])
    mlp_model.load_state_dict(mlp_state["model_state_dict"])

    _load_pretrained_neumf_weights(model, gmf_model, mlp_model)


def _load_pretrained_neumf_weights(
    neumf_model: NeuMF,
    gmf_model: GMF,
    mlp_model: MLP,
) -> None:
    with torch.no_grad():
        neumf_model.mf_user_embedding.weight.copy_(gmf_model.user_embedding.weight)
        neumf_model.mf_item_embedding.weight.copy_(gmf_model.item_embedding.weight)

        neumf_model.mlp_user_embedding.weight.copy_(mlp_model.user_embedding.weight)
        neumf_model.mlp_item_embedding.weight.copy_(mlp_model.item_embedding.weight)

        mlp_source_layers = [layer for layer in mlp_model.mlp if isinstance(layer, nn.Linear)]
        mlp_target_layers = [layer for layer in neumf_model.mlp if isinstance(layer, nn.Linear)]
        if len(mlp_source_layers) != len(mlp_target_layers):
            raise ValueError("MLP layer count mismatch while warm-starting NeuMF.")
        for source, target in zip(mlp_source_layers, mlp_target_layers, strict=False):
            target.weight.copy_(source.weight)
            target.bias.copy_(source.bias)

        gmf_weight = gmf_model.output.weight
        mlp_weight = mlp_model.output.weight
        combined_weight = torch.cat([gmf_weight, mlp_weight], dim=1)
        combined_bias = (gmf_model.output.bias + mlp_model.output.bias) * 0.5
        neumf_model.output.weight.copy_(combined_weight * 0.5)
        neumf_model.output.bias.copy_(combined_bias)
