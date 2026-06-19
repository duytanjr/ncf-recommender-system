"""CLI entrypoint for model training."""

from __future__ import annotations

from pathlib import Path

import typer
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from ncf_recommender.training.trainer import run_training

app = typer.Typer(help="Train NCF-family models with Hydra configs.")


@app.command()
def main(
    config_name: str = "config",
    overrides: list[str] = typer.Option([], "--set", help="Hydra overrides."),
) -> None:
    """Run training pipeline with a selected config profile."""
    cfg = _load_cfg(config_name=config_name, overrides=overrides)
    artifacts = run_training(cfg)
    typer.echo(
        "training completed "
        f"best_epoch={artifacts.best_epoch} "
        f"best_ckpt={artifacts.best_checkpoint} "
        f"last_ckpt={artifacts.last_checkpoint} "
        f"run_manifest={artifacts.run_manifest} "
        f"best_ndcg={artifacts.best_metrics['ndcg@k']:.4f}",
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
