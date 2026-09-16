from pathlib import Path
from unittest.mock import MagicMock, patch

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


def test_dataset_audit_command_is_thin(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    output = tmp_path / "manifest.json"
    with (
        patch(
            "clash_sos.interfaces.cli.main.audit_kaggle_v6_archive",
            return_value=object(),
        ) as audit,
        patch("clash_sos.interfaces.cli.main.write_dataset_manifest") as write,
    ):
        result = runner.invoke(
            cli_app,
            [
                "dataset",
                "audit-kaggle-v6",
                "--archive",
                str(archive),
                "--output",
                str(output),
                "--temp-directory",
                str(tmp_path),
                "--memory-limit",
                "512MB",
                "--threads",
                "1",
            ],
        )

    assert result.exit_code == 0
    assert result.stdout == f"{output}\n"
    audit.assert_called_once_with(
        archive,
        temp_directory=tmp_path,
        memory_limit="512MB",
        threads=1,
    )
    write.assert_called_once_with(audit.return_value, output)


def test_dataset_inspect_command_is_thin(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    manifest = MagicMock()
    manifest.model_dump_json.return_value = '{"source_id":"kaggle-v6"}'
    with (
        patch(
            "clash_sos.interfaces.cli.main.audit_kaggle_v6_archive",
            return_value=manifest,
        ) as audit,
        patch("clash_sos.interfaces.cli.main.write_dataset_manifest") as write,
    ):
        result = runner.invoke(
            cli_app,
            [
                "dataset",
                "inspect-kaggle-v6",
                "--archive",
                str(archive),
                "--temp-directory",
                str(tmp_path),
                "--memory-limit",
                "512MB",
                "--threads",
                "1",
            ],
        )

    assert result.exit_code == 0
    assert result.stdout == '{"source_id":"kaggle-v6"}\n'
    audit.assert_called_once_with(
        archive,
        temp_directory=tmp_path,
        memory_limit="512MB",
        threads=1,
    )
    write.assert_not_called()
    manifest.model_dump_json.assert_called_once_with(by_alias=True, indent=2)
