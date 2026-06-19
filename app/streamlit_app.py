"""Phase B Streamlit demo for NCF recommendation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
import json

import pandas as pd
import streamlit as st
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from ncf_recommender.data.datasets import load_legacy_implicit_dataset
from ncf_recommender.training.trainer import _build_model, _resolve_checkpoint_path, resolve_device

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w342"
PLACEHOLDER_IMAGE = "https://placehold.co/342x513?text=No+Poster"


@st.cache_resource(show_spinner=False)
def load_runtime(config_name: str, overrides: tuple[str, ...]) -> tuple[DictConfig, Any, torch.nn.Module, torch.device]:
    """Load config, dataset and model once per configuration."""
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=list(overrides))

    device = resolve_device(str(cfg.runtime.device))
    dataset = load_legacy_implicit_dataset(root=Path(cfg.data.root_dir), name=cfg.data.name)
    model = _build_model(cfg, dataset).to(device)

    ckpt = _resolve_checkpoint_path(cfg)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return cfg, dataset, model, device


@st.cache_data(show_spinner=False)
def load_movies_metadata(metadata_path: str) -> pd.DataFrame:
    """Load MovieLens movies.dat into dataframe."""
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"movies.dat not found: {metadata_path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="latin-1") as file:
        for line in file:
            parts = line.strip().split("::")
            if len(parts) != 3:
                continue
            item_id = int(parts[0])
            title = parts[1]
            genres = parts[2]
            year_match = re.search(r"\((\d{4})\)\s*$", title)
            year = int(year_match.group(1)) if year_match else None
            clean_title = re.sub(r"\s*\(\d{4}\)\s*$", "", title)
            records.append(
                {
                    "item_id": item_id,
                    "title": title,
                    "clean_title": clean_title,
                    "year": year,
                    "genres": genres,
                }
            )
    return pd.DataFrame(records)


def recommend_for_user(
    dataset: Any,
    model: torch.nn.Module,
    device: torch.device,
    user_id: int,
    top_k: int,
) -> list[tuple[int, float]]:
    interacted_items = {item for (user, item) in dataset.train_matrix.keys() if user == user_id}
    candidate_items = [item for item in range(dataset.num_items) if item not in interacted_items]
    if not candidate_items:
        return []

    users = torch.full((len(candidate_items),), fill_value=user_id, dtype=torch.long, device=device)
    items = torch.tensor(candidate_items, dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(users, items)
        scores = torch.sigmoid(logits).detach().cpu().tolist()

    ranked = sorted(zip(candidate_items, scores), key=lambda x: x[1], reverse=True)
    return [(int(item), float(score)) for item, score in ranked[:top_k]]


def user_history(dataset: Any, user_id: int) -> list[int]:
    return sorted(item for (user, item) in dataset.train_matrix.keys() if user == user_id)


@st.cache_data(show_spinner=False)
def fetch_tmdb_poster(api_key: str, title: str, year: int | None) -> str:
    """Fetch one TMDB poster URL for a movie title/year."""
    if not api_key:
        return PLACEHOLDER_IMAGE

    query = quote_plus(title)
    year_param = f"&year={year}" if year is not None else ""
    url = (
        "https://api.themoviedb.org/3/search/movie"
        f"?api_key={api_key}&query={query}{year_param}&include_adult=false"
    )
    request = Request(url, method="GET")
    try:
        with urlopen(request, timeout=6) as response:
            payload = json.loads(response.read().decode("utf-8"))
        results = payload.get("results", [])
        if not results:
            return PLACEHOLDER_IMAGE
        poster_path = results[0].get("poster_path")
        if not poster_path:
            return PLACEHOLDER_IMAGE
        return f"{TMDB_IMAGE_BASE}{poster_path}"
    except Exception:
        return PLACEHOLDER_IMAGE


def apply_feedback_rerank(df: pd.DataFrame) -> pd.DataFrame:
    """Apply in-session like/dislike bias and rerank."""
    likes = st.session_state.get("likes", set())
    dislikes = st.session_state.get("dislikes", set())
    adjusted = df.copy()
    bias = []
    for item_id in adjusted["item_id"]:
        if int(item_id) in likes:
            bias.append(0.05)
        elif int(item_id) in dislikes:
            bias.append(-0.05)
        else:
            bias.append(0.0)
    adjusted["adjusted_score"] = adjusted["score"] + pd.Series(bias)
    adjusted = adjusted.sort_values("adjusted_score", ascending=False).reset_index(drop=True)
    adjusted["rank"] = adjusted.index + 1
    return adjusted


def main() -> None:
    st.set_page_config(page_title="NCF Demo - Phase B", layout="wide")
    st.title("NCF Recommender Demo (Phase B)")
    st.caption("User-centric portal with posters, search/filter, and feedback loop.")

    if "likes" not in st.session_state:
        st.session_state["likes"] = set()
    if "dislikes" not in st.session_state:
        st.session_state["dislikes"] = set()

    with st.sidebar:
        st.header("Configuration")
        config_name = st.selectbox("Config", ["eval_gmf", "eval_mlp", "eval_neumf"], index=0)
        runtime_device = st.selectbox("Runtime Device", ["auto", "cpu", "cuda"], index=0)
        top_k = st.slider("Top-K", min_value=5, max_value=50, value=12, step=1)
        metadata_path = st.text_input("movies.dat path", value="configs/data/movies.dat")
        checkpoint_override = st.text_input(
            "Checkpoint override (optional)",
            value="",
            help="Leave empty to use config default checkpoint.",
        )
        try:
            default_tmdb_key = st.secrets.get("TMDB_API_KEY", os.getenv("TMDB_API_KEY", ""))
        except Exception:
            default_tmdb_key = os.getenv("TMDB_API_KEY", "")

        tmdb_api_key = st.text_input(
            "TMDB API Key",
            type="password",
            value=default_tmdb_key,
        )
        search_query = st.text_input("Search in recommendations", value="")

    overrides_list = [f"runtime.device={runtime_device}"]
    if checkpoint_override.strip():
        overrides_list.append(f"eval.checkpoint_path={checkpoint_override.strip()}")
    overrides = tuple(overrides_list)

    try:
        cfg, dataset, model, device = load_runtime(config_name=config_name, overrides=overrides)
        movies_df = load_movies_metadata(metadata_path)
    except Exception as exc:
        st.error(f"Failed to initialize app: {exc}")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Users", f"{dataset.num_users:,}")
    col2.metric("Items", f"{dataset.num_items:,}")
    col3.metric("Train Interactions", f"{dataset.train_matrix.nnz:,}")
    col4.metric("Device", str(device))

    st.markdown("### Recommend")
    user_id = st.number_input(
        "User ID",
        min_value=0,
        max_value=max(dataset.num_users - 1, 0),
        value=0,
        step=1,
    )

    if st.button("Generate Recommendations", type="primary"):
        recs = recommend_for_user(dataset=dataset, model=model, device=device, user_id=int(user_id), top_k=top_k)
        if not recs:
            st.warning("No candidate items found for this user.")
            st.stop()

        rec_df = pd.DataFrame(recs, columns=["item_id", "score"])
        rec_df = rec_df.merge(movies_df, on="item_id", how="left")
        rec_df["rank"] = rec_df.index + 1
        rec_df = apply_feedback_rerank(rec_df)

        if search_query.strip():
            rec_df = rec_df[rec_df["title"].fillna("").str.contains(search_query.strip(), case=False, regex=False)]

        st.markdown("### Top-K movie recommendations table")
        table_cols = ["rank", "item_id", "title", "genres", "score", "adjusted_score"]
        st.dataframe(rec_df[table_cols], use_container_width=True, hide_index=True)

        st.markdown("### Score Chart")
        chart_df = rec_df[["rank", "adjusted_score"]].set_index("rank")
        st.bar_chart(chart_df)

        st.markdown("### Recommended Movies")
        cols = st.columns(4)
        for idx, row in rec_df.iterrows():
            col = cols[idx % 4]
            title = row["title"] if pd.notna(row["title"]) else f"Item {int(row['item_id'])}"
            year = int(row["year"]) if pd.notna(row["year"]) else None
            clean_title = row["clean_title"] if pd.notna(row["clean_title"]) else title
            poster = fetch_tmdb_poster(tmdb_api_key, clean_title, year)
            with col:
                st.image(poster, use_container_width=True)
                st.markdown(f"**{title}**")
                st.caption(f"item_id={int(row['item_id'])} | score={row['adjusted_score']:.4f}")
                like_key = f"like_{int(row['item_id'])}"
                dislike_key = f"dislike_{int(row['item_id'])}"
                c1, c2 = st.columns(2)
                item_id_int = int(row['item_id'])

                is_liked = item_id_int in st.session_state["likes"]
                is_disliked = item_id_int in st.session_state["dislikes"]

                # Nút Like
                if c1.button(
                        "❤️ Like" if not is_liked else "💖 Liked",
                        key=f"like_{item_id_int}",
                        type="primary" if is_liked else "secondary"
                ):
                    if is_liked:  # Nếu đã like, bỏ like
                        st.session_state["likes"].discard(item_id_int)
                    else:  # Nếu chưa like, thêm like và bỏ dislike nếu có
                        st.session_state["likes"].add(item_id_int)
                        st.session_state["dislikes"].discard(item_id_int)
                    st.rerun()

                # Nút Dislike
                if c2.button(
                        "👎 Dislike" if not is_disliked else "🚫 Disliked",
                        key=f"dislike_{item_id_int}",
                        type="primary" if is_disliked else "secondary"
                ):
                    if is_disliked:  # Nếu đã dislike, bỏ dislike
                        st.session_state["dislikes"].discard(item_id_int)
                    else:  # Nếu chưa dislike, thêm dislike và bỏ like nếu có
                        st.session_state["dislikes"].add(item_id_int)
                        st.session_state["likes"].discard(item_id_int)
                    st.rerun()

        st.markdown("### User Interaction History (Train)")
        history = user_history(dataset=dataset, user_id=int(user_id))
        history_df = pd.DataFrame({"item_id": history}).merge(
            movies_df[["item_id", "title", "genres"]], on="item_id", how="left"
        )
        st.dataframe(history_df, use_container_width=True, hide_index=True, height=300)

        st.info(
            f"Model: `{cfg.model.name}` | Dataset: `{cfg.data.name}` | "
            f"Checkpoint: `{_resolve_checkpoint_path(cfg)}`"
        )


if __name__ == "__main__":
    main()
