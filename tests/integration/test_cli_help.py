from typer.testing import CliRunner

from ncf_recommender.cli.train import app


def test_train_cli_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["main", "--help"])
    assert result.exit_code == 0
    assert "Run training pipeline" in result.stdout
