from typer.testing import CliRunner

from clash_sos.interfaces.cli.main import app as cli_app
from clash_sos.interfaces.workers.main import app as worker_app

runner = CliRunner()


def test_cli_reports_version() -> None:
    result = runner.invoke(cli_app, ["version"])

    assert result.exit_code == 0
    assert result.stdout == "0.1.0\n"


def test_worker_reports_status() -> None:
    result = runner.invoke(worker_app, ["status"])

    assert result.exit_code == 0
    assert result.stdout == "ready\n"
