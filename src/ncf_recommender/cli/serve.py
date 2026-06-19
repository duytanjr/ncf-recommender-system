"""CLI entrypoint for serving model API."""

from __future__ import annotations

import uvicorn


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Serve FastAPI recommendation endpoint."""
    uvicorn.run("ncf_recommender.api.app:app", host=host, port=port, reload=False)


def run() -> None:
    import typer

    typer.run(main)


if __name__ == "__main__":
    run()
