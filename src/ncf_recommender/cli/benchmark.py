"""CLI entrypoint for model benchmarking."""

from __future__ import annotations

from pathlib import Path

import typer
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from ncf_recommender.training.benchmark import benchmark_model


def _load_cfg(config_name: str, overrides: list[str]) -> DictConfig:
    config_dir = Path(__file__).resolve().parents[3] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def main(
    config_name: str = "config",
    batch_size: int = 4096,
    steps: int = 200,
    warmup_steps: int = 20,
    overrides: list[str] = typer.Option([], "--set", help="Hydra overrides."),
) -> None:
    """Run micro-benchmark for inference performance."""
    cfg = _load_cfg(config_name=config_name, overrides=overrides)
    result = benchmark_model(
        cfg=cfg,
        batch_size=batch_size,
        steps=steps,
        warmup_steps=warmup_steps,
    )
    typer.echo(
        "benchmark completed "
        f"device={result['device']} "
        f"throughput={result['throughput_samples_per_s']:.1f} "
        f"latency_ms={result['latency_ms_per_batch']:.3f}",
    )


def run() -> None:
    typer.run(main)


if __name__ == "__main__":
    run()
