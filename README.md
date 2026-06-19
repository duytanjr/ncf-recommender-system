# ncf_recommender

Production-oriented reimplementation of Neural Collaborative Filtering (NCF) with modern PyTorch tooling.

This project modernizes the original `hexiangnan/neural_collaborative_filtering` codebase into a modular system for:
- training (`GMF`, `MLP`, `NeuMF`)
- evaluation (leave-one-out ranking)
- inference and demo apps
- model export and API serving
- reproducible experiment tracking

## Highlights
- Config-driven workflows via Hydra (`configs/`)
- End-to-end CLI for train/eval/infer/export/serve/benchmark
- `NeuMF` warm-start from pretrained `GMF` + `MLP`
- Runtime hardening: `cpu/cuda/auto`, AMP (`fp16`, `bf16`), resume training
- Metrics: `HR@K`, `NDCG@K`, `Recall@K`, `MRR@K`
- Streamlit demo (Phase A/B), FastAPI serving, TorchScript/ONNX export
- CI with lint/type-check/tests

## Repository Layout
- `src/ncf_recommender/models` - GMF/MLP/NeuMF model definitions
- `src/ncf_recommender/data` - legacy dataset loaders + sampling
- `src/ncf_recommender/training` - train loop, checkpointing, profiling, tracking
- `src/ncf_recommender/evaluation` - ranking metrics and protocols
- `src/ncf_recommender/inference` - recommendation + export helpers
- `src/ncf_recommender/api` - FastAPI app (`/healthz`, `/recommend`)
- `src/ncf_recommender/cli` - CLI entrypoints
- `configs` - Hydra config groups (`data/model/train/eval`)
- `app` - Streamlit demo
- `tests` - unit/integration tests
- `docs` - architecture, migration, session context

## Data
This repo is standalone and uses local data under `Data/` by default.

Supported dataset formats (legacy NCF):
- `*.train.rating`
- `*.test.rating`
- `*.test.negative`

## Installation
```bash
python -m venv .venv
. .venv/Scripts/activate  # PowerShell: .\\.venv\\Scripts\\Activate.ps1
pip install -e .[dev,loggers]
```

Optional extras:
- Demo UI: `pip install -e .[demo]`
- Serving: `pip install -e .[serving]`
- Export runtimes: `pip install -e .[export]`
- MLflow: `pip install -e .[mlops]`

## Quick Start
### 1) Train baselines
```bash
ncf-train --config-name train_gmf
ncf-train --config-name train_mlp
```

### 2) Train NeuMF (pretrained)
```bash
ncf-train --config-name train_neumf_pretrained
```

### 3) Evaluate
```bash
ncf-eval --config-name eval_neumf
```

### 4) Inference
```bash
ncf-infer 42 --top-k 10 --config-name eval_neumf
```

## Common Runtime Overrides
```bash
# CUDA + mixed precision
ncf-train --config-name train_neumf_pretrained --set runtime.device=cuda --set runtime.precision=fp16

# Resume from last checkpoint
ncf-train --config-name train_neumf_pretrained --set resume.enabled=true

# TensorBoard logging
ncf-train --config-name train_neumf_pretrained --set logging.tensorboard.enabled=true
```

## Benchmark and Profiling
```bash
# Training profiler traces
ncf-train --config-name train_gmf --set profiling.enabled=true

# Inference micro-benchmark
ncf-benchmark --config-name eval_gmf --batch-size 8192 --steps 200
```

## Export and Serve
```bash
# TorchScript
ncf-export --config-name eval_neumf --export-format torchscript --output-path artifacts/exports/neumf.ts

# ONNX
ncf-export --config-name eval_neumf --export-format onnx --output-path artifacts/exports/neumf.onnx

# API service
ncf-serve --host 0.0.0.0 --port 8000
```

API endpoints:
- `GET /healthz`
- `POST /recommend`

## Streamlit Demo
```bash
pip install -e .[demo]
streamlit run app/streamlit_app.py
```

Phase B supports:
- model/config switching
- checkpoint override
- poster rendering via TMDB API key
- search/filter
- in-session Like/Dislike reranking

For TMDB key (PowerShell):
```bash
$env:TMDB_API_KEY="your_key_here"
```

## Reproducibility and Tracking
- Best/last checkpoints under `artifacts/checkpoints`
- Run manifests under `artifacts/runs`
- Dataset fingerprinting enabled for traceability
- Optional W&B/MLflow integrations

## Development
```bash
pre-commit install
pytest -q
ruff check .
black --check .
mypy src
```

## Documentation
- Architecture: `docs/architecture.md`
- Migration guide: `docs/migration.md`
- Session context snapshot: `docs/session_context.md`

## Notes
- AMP safety is handled via logits output + `BCEWithLogitsLoss`.
- For CUDA runs, ensure driver and PyTorch CUDA wheel compatibility.
