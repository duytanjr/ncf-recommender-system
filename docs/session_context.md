# NCF Recommender — Context Snapshot

## 1) Project Overview
- **Tên dự án:** `ncf_recommender`
- **Mục tiêu:** Hiện đại hóa repo `neural_collaborative_filtering` thành hệ thống recommender production-ready (train/eval/infer/demo/serve/export).
- **Stack công nghệ chính:**
  - **Ngôn ngữ:** Python 3.11+
  - **ML:** PyTorch (hiện pin khả dụng thực tế: `torch>=2.6.0`)
  - **Config:** Hydra + OmegaConf
  - **CLI:** Typer
  - **Web Demo:** Streamlit (Phase A/B)
  - **API Serving:** FastAPI + Uvicorn
  - **Tracking/Logs:** TensorBoard, W&B (optional), MLflow (optional)
  - **Test/Quality:** pytest, mypy, ruff, black, GitHub Actions CI

## 2) Current Progress
### Hoàn thiện 100%
- Re-architecture thành package `src/ncf_recommender/*` (module hóa rõ ràng).
- Huấn luyện/evaluate/inference cho **GMF / MLP / NeuMF**.
- **NeuMF pretrained**: warm-start từ checkpoint GMF + MLP.
- Leave-one-out evaluation + metrics: **HR@K, NDCG@K, Recall@K, MRR**.
- Checkpointing: `best` + `last`; early stopping; resume train.
- Runtime hardening: device policy (`auto/cpu/cuda`), precision (`fp32/fp16/bf16`), AMP.
- Benchmark/profiling cơ bản.
- Export model: TorchScript/ONNX.
- API serving `/healthz`, `/recommend`.
- Streamlit demo:
  - **Phase A:** bảng top-K + score + chart + history.
  - **Phase B:** poster TMDB + search/filter + like/dislike rerank session.
- Repo đã **standalone dữ liệu** (`Data/` nội bộ), không còn phụ thuộc runtime repo cũ.
- Test suite đã pass trước đó (`pytest` pass toàn bộ trên môi trường user).

### Đang dở / cần xác nhận thêm
- Chưa có vòng benchmark chính thức có số liệu chuẩn (latency/throughput theo model/device) lưu thành báo cáo.
- Chưa có test e2e cho API + Streamlit.
- Chưa có pipeline retraining tự động từ feedback (feedback hiện là session-level demo).

## 3) Key Logic & Structure (Core Files)
- `src/ncf_recommender/training/trainer.py`  
  Vòng train/eval chính, AMP, resume, checkpoint, early-stopping, profiling, tracking hooks.
- `src/ncf_recommender/models/gmf.py`  
  Định nghĩa model GMF (logits output).
- `src/ncf_recommender/models/mlp.py`  
  Định nghĩa model MLP (logits output).
- `src/ncf_recommender/models/neumf.py`  
  Định nghĩa model NeuMF + dùng cho warm-start pretrained flow.
- `src/ncf_recommender/evaluation/metrics.py`  
  Các metric ranking cốt lõi.
- `src/ncf_recommender/evaluation/protocols.py`  
  Leave-one-out evaluation protocol.
- `src/ncf_recommender/data/datasets.py`  
  Load data legacy (`*.train.rating`, `*.test.rating`, `*.test.negative`) + fingerprint.
- `src/ncf_recommender/inference/service.py`  
  Top-K inference (+ score), history/stats helper.
- `src/ncf_recommender/inference/export.py`  
  Export TorchScript/ONNX.
- `src/ncf_recommender/api/app.py`  
  FastAPI serving endpoints.
- `app/streamlit_app.py`  
  Demo UI Phase B.
- `configs/config.yaml` + `configs/{model,train,eval,data}/*.yaml`  
  Toàn bộ cấu hình runtime/train/eval/model/data.
- `pyproject.toml`  
  Dependency + entrypoints CLI.
- `.github/workflows/ci.yml`  
  CI lint/type/test matrix.

## 4) Pending Tasks (To-do)
1. Chốt benchmark report chuẩn (CPU/GPU, GMF/MLP/NeuMF, batch sizes).
2. Thêm test integration cho:
   - API `/recommend`
   - Export ONNX/TorchScript load-back check
   - Streamlit smoke test.
3. Tách metadata layer chính thức cho MovieLens (`item_id -> title/year/genres`) thành module riêng (thay vì chỉ trong app).
4. Chuẩn hóa quản lý secret:
   - TMDB key qua `.streamlit/secrets.toml` hoặc env var.
   - Rotate key nếu đã từng lộ.
5. (Optional production) Thêm observability sâu hơn (structured logs + request latency metrics cho API).
6. (Optional MLOps) Thiết kế feedback persistence + retrain job pipeline.

## 5) Technical Constraints
- **Loss/AMP:** Model output là **logits**, training dùng `BCEWithLogitsLoss` (không dùng `sigmoid + BCELoss` khi autocast).
- **Torch version thực tế:** dùng `torch>=2.6.0` để tương thích wheel CUDA đang khả dụng (`cu124`); trước đó `>=2.7.0` không resolve được.
- **GPU runtime:** `runtime.device=cuda` sẽ fail nếu CUDA không khả dụng hoặc driver/wheel mismatch.
- **Data path mặc định:** đã chuyển sang `root_dir: Data` trong:
  - `configs/data/ml1m.yaml`
  - `configs/data/pinterest20.yaml`
- **Checkpoint compatibility:** Config model phải khớp kiến trúc checkpoint (`gmf` vs `mlp` vs `neumf`).
- **TMDB key:** không hard-code vào code; ưu tiên `st.secrets`/env (`TMDB_API_KEY`).
- **Known issue đã xử lý:** `weights_only` load behavior của PyTorch mới; code đã load checkpoint phù hợp (`weights_only=False` ở các điểm cần thiết).
