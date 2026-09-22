from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from clash_sos.application.dataset_prepare import KaggleV6PrepareError
from clash_sos.application.dataset_staging import StagingConfig
from clash_sos.application.model_train import KaggleV6ModelTrainError
from clash_sos.interfaces.cli.main import app as cli_app
from clash_sos.interfaces.cli.main import parse_aware_datetime
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


def test_dataset_prepare_command_is_thin(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    destination = tmp_path / "published"
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    temp_directory = tmp_path / "tmp"
    raw_manifest = tmp_path / "raw.json"
    with patch(
        "clash_sos.interfaces.cli.main.prepare_kaggle_v6_dataset",
        return_value=destination,
    ) as prepare:
        result = runner.invoke(
            cli_app,
            [
                "dataset",
                "prepare-kaggle-v6",
                "--archive",
                str(archive),
                "--destination",
                str(destination),
                "--staging-workspace",
                str(staging),
                "--output-workspace",
                str(output),
                "--temp-directory",
                str(temp_directory),
                "--raw-manifest",
                str(raw_manifest),
                "--train-end",
                "2026-06-10T00:00:00+00:00",
                "--validation-end",
                "2026-06-20T00:00:00+00:00",
                "--memory-limit",
                "512MB",
                "--threads",
                "1",
                "--chunk-size",
                "1024",
                "--batch-rows",
                "8",
                "--player-seed",
                "3",
                "--player-train-max",
                "0.6",
                "--player-validation-max",
                "0.8",
                "--dataset-version",
                "test-v1",
            ],
        )

    assert result.exit_code == 0
    assert result.stdout == f"{destination}\n"
    prepare.assert_called_once_with(
        archive,
        destination,
        staging_workspace=staging,
        output_workspace=output,
        temp_directory=temp_directory,
        train_end=datetime(2026, 6, 10, tzinfo=UTC),
        validation_end=datetime(2026, 6, 20, tzinfo=UTC),
        config=StagingConfig(
            memory_limit="512MB",
            threads=1,
            chunk_size=1024,
            batch_rows=8,
        ),
        raw_manifest_path=raw_manifest,
        player_seed=3,
        player_train_max=0.6,
        player_validation_max=0.8,
        dataset_version="test-v1",
    )


def test_dataset_prepare_command_requires_temporal_cutovers() -> None:
    result = runner.invoke(cli_app, ["dataset", "prepare-kaggle-v6"])

    assert result.exit_code != 0


def test_dataset_prepare_command_surfaces_existing_version() -> None:
    with patch(
        "clash_sos.interfaces.cli.main.prepare_kaggle_v6_dataset",
        side_effect=KaggleV6PrepareError("published dataset version already exists"),
    ):
        result = runner.invoke(
            cli_app,
            [
                "dataset",
                "prepare-kaggle-v6",
                "--train-end",
                "2026-06-10T00:00:00+00:00",
                "--validation-end",
                "2026-06-20T00:00:00+00:00",
            ],
        )

    assert result.exit_code != 0
    assert result.exception is not None


def test_dataset_verify_command_is_thin(tmp_path: Path) -> None:
    dataset = tmp_path / "published"
    temp_directory = tmp_path / "tmp"
    with patch("clash_sos.interfaces.cli.main.verify_kaggle_v6_dataset") as verify:
        result = runner.invoke(
            cli_app,
            [
                "dataset",
                "verify-kaggle-v6",
                "--dataset",
                str(dataset),
                "--temp-directory",
                str(temp_directory),
                "--memory-limit",
                "512MB",
                "--threads",
                "1",
                "--chunk-size",
                "1024",
            ],
        )

    assert result.exit_code == 0
    assert result.stdout == f"{dataset}\n"
    verify.assert_called_once_with(
        dataset,
        config=StagingConfig(memory_limit="512MB", threads=1, chunk_size=1024),
        temp_directory=temp_directory,
    )


def test_parse_aware_datetime_requires_timezone() -> None:
    with pytest.raises(typer.BadParameter, match="timezone-aware"):
        parse_aware_datetime("2026-06-10T00:00:00")


def test_model_train_command_is_thin(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    destination = tmp_path / "model"
    output = tmp_path / "output"
    temp_directory = tmp_path / "tmp"
    with patch(
        "clash_sos.interfaces.cli.main.train_card_pair_model",
        return_value=destination,
    ) as train:
        result = runner.invoke(
            cli_app,
            [
                "model",
                "train",
                "--dataset",
                str(dataset),
                "--destination",
                str(destination),
                "--output-workspace",
                str(output),
                "--temp-directory",
                str(temp_directory),
                "--memory-limit",
                "512MB",
                "--threads",
                "1",
                "--chunk-size",
                "1024",
                "--smoothing-alpha",
                "2.0",
                "--mirror-seed",
                "4",
                "--model-version",
                "test-model-v1",
                "--promoted-model",
                "card_pair",
            ],
        )

    assert result.exit_code == 0
    assert result.stdout == f"{destination}\n"
    train.assert_called_once_with(
        dataset,
        destination,
        output_workspace=output,
        temp_directory=temp_directory,
        config=StagingConfig(memory_limit="512MB", threads=1, chunk_size=1024),
        smoothing_alpha=2.0,
        mirror_seed=4,
        model_version="test-model-v1",
        progress=ANY,
    )


def test_model_train_defaults_to_lightgbm(tmp_path: Path) -> None:
    destination = tmp_path / "model"
    with patch(
        "clash_sos.application.model_train_lgbm.train_lightgbm_model",
        return_value=destination,
    ) as train:
        result = runner.invoke(
            cli_app,
            ["model", "train", "--destination", str(destination)],
        )

    assert result.exit_code == 0
    assert result.stdout == f"{destination}\n"
    train.assert_called_once()
    assert train.call_args.kwargs["model_version"] == "kaggle-v6-ranked16-lightgbm-v3"
    assert train.call_args.kwargs["num_threads"] == 4
    assert train.call_args.kwargs["config"] == StagingConfig(
        memory_limit="1GB", threads=1, chunk_size=8 * 1024 * 1024
    )


def test_model_train_command_dispatches_card_log_odds(tmp_path: Path) -> None:
    destination = tmp_path / "model"
    with patch(
        "clash_sos.interfaces.cli.main.train_matchup_baseline",
        return_value=destination,
    ) as train:
        result = runner.invoke(
            cli_app,
            [
                "model",
                "train",
                "--promoted-model",
                "card_log_odds",
                "--destination",
                str(destination),
                "--model-version",
                "test-model-v1",
            ],
        )

    assert result.exit_code == 0
    train.assert_called_once()
    assert train.call_args.kwargs["model_version"] == "test-model-v1"
    assert train.call_args.args[1] == destination


def test_model_train_lgbm_command_dispatches(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    destination = tmp_path / "model"
    output = tmp_path / "output"
    temp_directory = tmp_path / "tmp"
    with patch(
        "clash_sos.application.model_train_lgbm.train_lightgbm_model",
        return_value=destination,
    ) as train:
        result = runner.invoke(
            cli_app,
            [
                "model",
                "train-lgbm",
                "--dataset",
                str(dataset),
                "--destination",
                str(destination),
                "--output-workspace",
                str(output),
                "--temp-directory",
                str(temp_directory),
                "--memory-limit",
                "512MB",
                "--threads",
                "1",
                "--chunk-size",
                "1024",
                "--smoothing-alpha",
                "2.0",
                "--mirror-seed",
                "4",
                "--model-version",
                "test-lgbm-v1",
            ],
        )

    assert result.exit_code == 0
    assert result.stdout == f"{destination}\n"
    train.assert_called_once_with(
        dataset,
        destination,
        output_workspace=output,
        temp_directory=temp_directory,
        config=StagingConfig(memory_limit="512MB", threads=1, chunk_size=1024),
        smoothing_alpha=2.0,
        mirror_seed=4,
        model_version="test-lgbm-v1",
        num_threads=1,
        progress=ANY,
    )


def test_model_train_command_surfaces_existing_version() -> None:
    with patch(
        "clash_sos.application.model_train_lgbm.train_lightgbm_model",
        side_effect=KaggleV6ModelTrainError("published model version already exists"),
    ):
        result = runner.invoke(cli_app, ["model", "train"])

    assert result.exit_code != 0
    assert result.exception is not None
