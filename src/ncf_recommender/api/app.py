"""FastAPI serving app for top-k recommendation."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from hydra import compose, initialize_config_dir
from pydantic import BaseModel, Field

from ncf_recommender.inference.service import recommend_top_k


class RecommendRequest(BaseModel):
    user_id: int = Field(..., ge=0)
    top_k: int = Field(default=10, ge=1, le=100)
    config_name: str = "config"
    overrides: list[str] = []


class RecommendResponse(BaseModel):
    recommended_items: list[int]


app = FastAPI(title="NCF Recommender API", version="0.1.0")


def _load_cfg(config_name: str, overrides: list[str]):
    config_dir = Path(__file__).resolve().parents[3] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest) -> RecommendResponse:
    try:
        cfg = _load_cfg(payload.config_name, payload.overrides)
        items = recommend_top_k(cfg=cfg, user_id=payload.user_id, top_k=payload.top_k)
        return RecommendResponse(recommended_items=items)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
