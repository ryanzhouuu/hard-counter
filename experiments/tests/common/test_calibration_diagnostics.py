import json
from dataclasses import replace
from math import exp, log
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from experiments.common import calibration
from experiments.common.calibration_diagnostics import fit_recorded_temperature
from experiments.common.data_access import ResearchRow
from experiments.common.predictions import read_predictions
from experiments.matchup_features.model import ResearchModel
from experiments.tests.common.research_fixture import rows


def test_records_known_temperature_without_changing_raw_scores(tmp_path: Path) -> None:
    population = tuple(replace(r, label=int(i % 5 != 0)) for i, r in enumerate(rows(20)))
    logits = np.full(20, 2 * log(4))
    fitted = fit_recorded_temperature(tmp_path, population, logits, output="matchup")
    assert fitted.temperature == pytest.approx(2, rel=1e-5)
    raw = read_predictions(tmp_path / "matchup-calibration-raw.json")
    assert [r.row_key for r in raw] == [r.key for r in population]
    assert all(r.logit == 2 * log(4) and r.probability == r.raw_probability for r in raw)
    diagnostics = json.loads((tmp_path / "matchup-calibration-diagnostics.json").read_bytes())
    assert diagnostics["status"] == "complete"
    assert diagnostics["fit"]["temperature"] == fitted.temperature
    assert diagnostics["raw_loss"] == pytest.approx(fitted.raw_loss)
    assert diagnostics["log_temperature_bounds"] == [-4, 4]
    assert diagnostics["loss_curve"][0]["temperature"] == exp(-4)
    assert diagnostics["loss_curve"][-1]["temperature"] == exp(4)


@pytest.mark.parametrize("aligned", [False, True])
def test_boundary_preserves_diagnostics_and_rejects_fit(tmp_path: Path, aligned: bool) -> None:
    population = tuple(replace(r, label=1) for r in rows(20))
    logits = np.full(20, 1.0 if aligned else -1.0)
    with pytest.raises(ValueError, match="boundary"):
        fit_recorded_temperature(tmp_path, population, logits, output="actual")
    diagnostics = json.loads((tmp_path / "actual-calibration-diagnostics.json").read_bytes())
    assert diagnostics["status"] == "failed" and "boundary" in diagnostics["reason"]
    assert diagnostics["mean_signed_logit"] == (1 if aligned else -1)
    assert diagnostics["inverse_temperature_derivative_at_zero"] == (-0.5 if aligned else 0.5)
    losses = [point["loss"] for point in diagnostics["loss_curve"]]
    assert losses == sorted(losses, reverse=not aligned)
    assert len(read_predictions(tmp_path / "actual-calibration-raw.json")) == 20
    assert not (tmp_path / "calibration.json").exists()


def test_unidentified_temperature_keeps_raw_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unidentified"):
        fit_recorded_temperature(tmp_path, rows(), np.zeros(24), output="matchup")
    diagnostics = json.loads((tmp_path / "matchup-calibration-diagnostics.json").read_bytes())
    assert diagnostics["status"] == "failed" and "unidentified" in diagnostics["reason"]
    assert all(point["loss"] == log(2) for point in diagnostics["loss_curve"])


def test_optimizer_error_remains_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def failed(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(success=False, x=0.0, fun=0.0)

    monkeypatch.setattr(calibration, "minimize_scalar", failed)
    with pytest.raises(ValueError, match="optimizer failed"):
        fit_recorded_temperature(tmp_path, rows(), np.ones(24), output="matchup")
    diagnostics = json.loads((tmp_path / "matchup-calibration-diagnostics.json").read_bytes())
    assert diagnostics["status"] == "failed" and "optimizer failed" in diagnostics["reason"]


def test_failed_run_inventories_evidence_without_publishing_calibrated_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments.common import scoring
    from experiments.common.artifacts import load_run
    from experiments.common.contracts import StudyConfig, Variant
    from experiments.common.run import run_variant
    from experiments.common.session import prepare_session

    config = StudyConfig(study_id="response-cycle", variants=(Variant(variant_id="A0"),))
    inputs = prepare_session(
        config,
        tmp_path / "inputs",
        synthetic=True,
        dataset=None,
        protocol=None,
        schema=None,
        cache=None,
        mechanics=None,
        row_cap=128,
    )

    def misaligned_logits(
        model: ResearchModel, population: tuple[ResearchRow, ...], features: np.ndarray
    ) -> np.ndarray:
        return np.asarray([0.1 * (1 - 2 * r.label) for r in population])

    monkeypatch.setattr(scoring, "predict_logits", misaligned_logits)
    destination = tmp_path / "failed"
    run_variant(
        Path.cwd(),
        destination,
        config,
        config.variants[0],
        inputs.access,
        inputs.population,
        inputs.catalog,
        inputs.schema,
        seed=0,
        penalty=0,
        phase="smoke",
    )
    manifest = load_run(destination)
    assert manifest.status == "failed" and not manifest.eligible_for_comparison
    assert "boundary" in manifest.failures[0]
    inventory = {record.path for record in manifest.outputs}
    assert {
        "fit-diagnostics.json",
        "matchup-calibration-raw.json",
        "matchup-calibration-diagnostics.json",
        "failure.json",
    } <= inventory
    assert not {"checkpoint.pt", "development.json", "calibration.json"} & inventory
    fit = json.loads((destination / "fit-diagnostics.json").read_bytes())
    assert len(fit["selection_training_losses"]) == len(fit["watch_losses"])
    assert len(fit["refit_training_losses"]) == fit["selected_epoch"]
    assert all(
        np.isfinite(fit[name]).all()
        for name in ("selection_training_losses", "watch_losses", "refit_training_losses")
    )
