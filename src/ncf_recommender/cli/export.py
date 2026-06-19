"""CLI entrypoint for exporting trained models."""

from __future__ import annotations

from pathlib import Path

import typer
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from ncf_recommender.inference.export import export_model


def _load_cfg(config_name: str, overrides: list[str]) -> DictConfig:
    config_dir = Path(__file__).resolve().parents[3] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def main(
    config_name: str = "config",
    output_path: str = "artifacts/exports/model.ts",
    export_format: str = "torchscript",
    overrides: list[str] = typer.Option([], "--set", help="Hydra overrides."),
) -> None:
    """Export checkpoint to TorchScript or ONNX."""
    cfg = _load_cfg(config_name=config_name, overrides=overrides)
    out = export_model(
        cfg=cfg,
        output_path=Path(output_path),
        export_format=export_format,
    )
    typer.echo(f"export completed artifact={out}")


def run() -> None:
    typer.run(main)


if __name__ == "__main__":
    run()
