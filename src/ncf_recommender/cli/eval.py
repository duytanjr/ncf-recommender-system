"""CLI entrypoint for offline evaluation."""

from __future__ import annotations

from pathlib import Path

import typer
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from ncf_recommender.training.trainer import run_evaluation

app = typer.Typer(help="Evaluate NCF-family checkpoints.")


@app.command()
def main(
    config_name: str = "config",
    overrides: list[str] = typer.Option([], "--set", help="Hydra overrides."),
) -> None:
    """Run evaluation pipeline with a selected config profile."""
    cfg = _load_cfg(config_name=config_name, overrides=overrides)
    metrics = run_evaluation(cfg)
    typer.echo(
        "eval completed "
        f"HR@{int(cfg.eval.top_k)}={metrics['hr@k']:.4f} "
        f"NDCG@{int(cfg.eval.top_k)}={metrics['ndcg@k']:.4f} "
        f"MRR@{int(cfg.eval.top_k)}={metrics['mrr@k']:.4f}",
    )


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
