# Architecture Overview

## Runtime layers
- `data`: parses legacy implicit-feedback files and builds training/evaluation inputs.
- `models`: GMF, MLP, NeuMF torch modules.
- `training`: orchestration loop, checkpointing, early stopping, optional experiment loggers.
- `evaluation`: leave-one-out ranking protocol and metrics.
- `inference`: top-k item retrieval with trained checkpoints.
- `api`: FastAPI service layer for online recommendation.
- `cli`: typed Typer entrypoints for train/eval/infer.

## Training lifecycle
1. Load dataset split files from configured dataset root.
2. Build model from `configs/model/*`.
3. Per epoch: generate point-wise negative samples, optimize BCE objective.
4. Run leave-one-out ranking evaluation.
5. Save `last` checkpoint every epoch, update `best` checkpoint on monitor improvement.
6. Stop early on configured patience.

## Runtime policy
- Device policy: `runtime.device` in `{auto,cpu,cuda}`.
- Precision policy: `runtime.precision` in `{fp32,fp16,bf16}`.
- Optional `torch.compile` via `runtime.compile_model`.
- Resume training via `resume.enabled` and optional `resume.checkpoint_path`.

## MLOps metadata
- Run manifests are emitted to `artifacts/runs`.
- Manifest includes config snapshot, best metrics, checkpoint paths, dataset fingerprint.
- Optional MLflow integration logs params, metrics, and run artifacts.

## Checkpoint contract
Each checkpoint stores:
- `epoch`
- `model_state_dict`
- `optimizer_state_dict`
- `metrics`
- `config` (plain dict)

This contract is shared across GMF/MLP/NeuMF to keep evaluation and inference generic.
