import json
from pathlib import Path

import numpy as np
import pytest
from experiments.common.artifacts import load_run
from experiments.common.checkpoints import load_checkpoint
from experiments.common.cli import main
from experiments.common.predictions import read_predictions
from experiments.common.synthetic import synthetic_smoke_population


@pytest.mark.parametrize(
    "study",
    ["response-cycle", "player-adjustment", "form-mechanics", "tower-mechanics", "higher-order"],
)
def test_all_studies_publish_reload_and_preserve_probabilities(tmp_path: Path, study: str) -> None:
    main(
        study,
        [
            "smoke",
            "--synthetic",
            "--output",
            str(tmp_path / "data"),
            "--models",
            str(tmp_path / "models"),
            "--run-id",
            "integration",
        ],
    )
    access, _, _ = synthetic_smoke_population()
    rows = access.read("development", "compare")
    directories = sorted((tmp_path / "models" / study / "integration").iterdir())
    for directory in directories:
        manifest = load_run(directory)
        assert manifest.status == "complete" and not manifest.eligible_for_comparison
        checkpoint = load_checkpoint(directory / "checkpoint.pt")
        expected = read_predictions(directory / "development.json")
        assert np.allclose(checkpoint.logits(rows), [r.logit for r in expected], atol=1e-7)
        report = json.loads((directory / "report.json").read_bytes())
        assert report["finite_gradients"] and report["swap_error"] < 1e-5
        assert (directory / "feature-cache" / "identity.json").exists()
        if study == "player-adjustment":
            diagnostic = report["player_diagnostics"]
            assert diagnostic["training"]["players"]
            assert len(diagnostic["evaluation_support"]) == len(rows)
            assert sum(diagnostic["support_bin_counts"]["prior_history"]) == len(rows) * 2
            assert sum(diagnostic["support_bin_counts"]["deck_switching"]) == len(rows) * 2
            assert diagnostic["support_bins"] == {
                "prior_history_edges": [1, 10],
                "deck_switching_edges": [1, 2],
            }
            assert diagnostic["actual_outcome_metrics"]["metrics"]["row_count"] == len(rows)
            assert sum(item["count"] for item in diagnostic["matchup_residuals"]) == len(rows) * 2
            assert all(
                item["interpretation"] == "descriptive-development-residual"
                for item in diagnostic["actual_outcome_residuals"]
            )
        if study == "player-adjustment" and checkpoint.metadata.variant.nuisance != "none":
            actual = read_predictions(directory / "actual-development.json")
            assert np.allclose(
                checkpoint.logits(rows, actual=True), [r.logit for r in actual], atol=1e-7
            )
            calibration = access.read("calibration", "calibrate")
            actual_calibration = read_predictions(directory / "actual-calibration.json")
            assert np.allclose(
                checkpoint.logits(calibration, actual=True),
                [r.logit for r in actual_calibration],
                atol=1e-7,
            )
            nuisance = report["player_diagnostics"]["nuisance_shrinkage"]
            assert nuisance["penalty"] == pytest.approx(0.0001)
            assert nuisance["penalty_value"] >= 0


def test_smoke_config_cannot_enter_development_or_reporting(tmp_path: Path) -> None:
    for command in ("run-development", "compare", "confirm"):
        with pytest.raises(PermissionError):
            main("response-cycle", [command, "--synthetic", "--output", str(tmp_path)])
