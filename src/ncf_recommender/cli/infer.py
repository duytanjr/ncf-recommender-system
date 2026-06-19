"""CLI entrypoint for top-k inference."""

from __future__ import annotations

from pathlib import Path

import typer
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from ncf_recommender.inference.service import recommend_top_k

app = typer.Typer(help="Generate top-k recommendations.")


@app.command()
def main(
    user_id: int,
    top_k: int = 10,
    config_name: str = "config",
    overrides: list[str] = typer.Option([], "--set", help="Hydra overrides."),
) -> None:
    """Run inference for one user."""
    cfg = _load_cfg(config_name=config_name, overrides=overrides)
    recommendations = recommend_top_k(cfg=cfg, user_id=user_id, top_k=top_k)
    typer.echo(f"recommended_items={recommendations}")


def _load_cfg(config_name: str, overrides: list[str]) -> DictConfig:
    config_dir = Path(__file__).resolve().parents[3] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def run() -> None:
    """Console script launcher."""
    typer.run(main)


if __name__ == "__main__":
    run()
