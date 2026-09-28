"""Check attention CLI forwarding, optional runtime errors, and real tiny fits."""

from pathlib import Path
from unittest.mock import patch

import pytest
from attention_cache_fixture import FORMS, LOSE, WIN, write_cache_dataset
from typer.testing import CliRunner

from clash_sos.application.model_predict import predict_matchup
from clash_sos.domain.canonical_dataset import canonical_json_bytes
from clash_sos.infrastructure.ml.attention_runtime import NeuralRuntimeUnavailable
from clash_sos.interfaces.cli.main import app

runner = CliRunner()


def test_train_attention_cli_forwards_validated_options(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"
    with patch(
        "clash_sos.application.model_train_attention.train_attention_artifact",
        return_value=destination,
    ) as train:
        result = runner.invoke(
            app,
            [
                "model",
                "train-attention",
                "--protocol",
                str(tmp_path / "protocol.json"),
                "--dataset",
                str(tmp_path / "dataset"),
                "--destination",
                str(destination),
                "--cache-directory",
                str(tmp_path / "cache"),
                "--output-workspace",
                str(tmp_path / "output"),
                "--network-config",
                str(tmp_path / "network.json"),
                "--catalog",
                str(tmp_path / "catalog.json"),
                "--attributes",
                str(tmp_path / "attributes.json"),
                "--model-version",
                "attention-test-v1",
                "--seed",
                "11",
                "--device",
                "cpu",
                "--batch-size",
                "32",
                "--max-epochs",
                "4",
                "--patience",
                "2",
                "--learning-rate",
                "0.002",
                "--weight-decay",
                "0.001",
                "--gradient-clip-norm",
                "0.5",
                "--time-limit-seconds",
                "30",
                "--memory-limit",
                "256MB",
                "--threads",
                "1",
                "--chunk-size",
                "1024",
                "--batch-rows",
                "100",
            ],
        )
    assert result.exit_code == 0
    assert result.stdout == f"{destination}\n"
    args, options = train.call_args
    assert args == (tmp_path / "dataset", destination)
    assert options["protocol_path"] == tmp_path / "protocol.json"
    assert options["cache_directory"] == tmp_path / "cache"
    assert options["output_workspace"] == tmp_path / "output"
    assert options["network_config_path"] == tmp_path / "network.json"
    assert options["catalog_path"] == tmp_path / "catalog.json"
    assert options["attributes_path"] == tmp_path / "attributes.json"
    assert options["model_version"] == "attention-test-v1"
    assert options["fit_config"].model_dump()["batch_size"] == 32
    assert options["fit_config"].model_dump()["learning_rate"] == 0.002
    assert options["staging_config"].chunk_size == 1024


def test_train_attention_cli_reports_missing_runtime_and_invalid_options(tmp_path: Path) -> None:
    with patch(
        "clash_sos.interfaces.cli.model_attention.initialize_attention_runtime",
        side_effect=NeuralRuntimeUnavailable("install the ml extra"),
    ):
        missing = runner.invoke(app, ["model", "train-attention", "--protocol", "missing.json"])
    assert missing.exit_code != 0
    assert "install the ml extra" in missing.output
    invalid = runner.invoke(
        app, ["model", "train-attention", "--protocol", "missing.json", "--batch-size", "0"]
    )
    assert invalid.exit_code != 0
    assert "batch_size" in invalid.output
    no_protocol = runner.invoke(app, ["model", "train-attention"])
    assert no_protocol.exit_code == 2


def test_train_attention_cli_fits_and_reloads_tiny_data(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _, protocol = write_cache_dataset(dataset)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_bytes(canonical_json_bytes(protocol.model_dump(mode="python")))
    destination = tmp_path / "artifact"
    command = [
        "model",
        "train-attention",
        "--protocol",
        str(protocol_path),
        "--dataset",
        str(dataset),
        "--destination",
        str(destination),
        "--cache-directory",
        str(tmp_path / "cache"),
        "--output-workspace",
        str(tmp_path / "output"),
        "--model-version",
        "attention-tiny-v1",
        "--device",
        "cpu",
        "--batch-size",
        "1",
        "--max-epochs",
        "1",
        "--patience",
        "1",
        "--memory-limit",
        "256MB",
        "--chunk-size",
        "1024",
        "--batch-rows",
        "1",
    ]
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    assert result.stdout == f"{destination}\n"
    assert "selected epoch 1" in result.stderr
    assert (destination / "manifest.json").is_file()
    first = tuple(f"{card}:{form}" for card, form in zip(WIN, FORMS, strict=True))
    second = tuple(f"{card}:{form}" for card, form in zip(LOSE, FORMS, strict=True))
    prediction = predict_matchup(destination, first, second, balance_era_id="2026-06")
    swapped = predict_matchup(destination, second, first, balance_era_id="2026-06")
    equal = predict_matchup(destination, first, first, balance_era_id="2026-06")
    assert prediction.provenance is not None
    assert prediction.provenance.model_version == "attention-tiny-v1"
    assert prediction.side_a_win_probability is not None
    assert swapped.side_a_win_probability == pytest.approx(1 - prediction.side_a_win_probability)
    assert equal.side_a_win_probability == pytest.approx(0.5)
    with pytest.raises(ValueError, match="balance era ID"):
        predict_matchup(destination, first, second)
    with pytest.raises(ValueError, match="does not cover"):
        predict_matchup(destination, first, second, balance_era_id="2026-07")
    with pytest.raises(ValueError, match="exactly eight"):
        predict_matchup(destination, first[:7], second, balance_era_id="2026-06")
    refused = runner.invoke(app, command)
    assert refused.exit_code != 0
    assert (destination / "manifest.json").is_file()
