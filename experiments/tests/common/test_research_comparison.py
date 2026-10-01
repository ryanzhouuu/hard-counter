import json
from dataclasses import replace
from pathlib import Path

import pytest
from experiments.common.artifacts import file_record, load_run
from experiments.common.comparison import comparison_report
from experiments.common.ensembles import FrozenEnsemble, calibrated_ensemble, publish_comparison
from experiments.common.predictions import read_predictions
from experiments.tests.common.comparison_fixture import comparison_fixture


def test_comparison_averages_raw_seed_probabilities_and_freezes_assets(tmp_path: Path) -> None:
    config, models, reports, _ = comparison_fixture(tmp_path)
    result = comparison_report(config, {"A1": 0.1}, models, reports)
    asset = FrozenEnsemble.model_validate_json((reports / "ensemble-A1.json").read_bytes())
    raw = tuple(
        read_predictions(models / f"A1-seed{seed}-penalty0.1" / "calibration.json")
        for seed in config.seeds
    )
    ensemble = read_predictions(reports / "ensemble-A1-calibration.json")
    assert ensemble[0].raw_probability == pytest.approx(
        sum(row[0].raw_probability for row in raw if row[0].raw_probability is not None) / len(raw)
    )
    assert asset.temperature > 0
    assert len(asset.seed_runs) == 3
    assert all(Path(seed.directory).is_absolute() for seed in asset.seed_runs)
    comparisons = result["comparisons"]
    assert isinstance(comparisons, dict)
    assert "A1 minus A0" in comparisons
    payload = json.loads((reports / "comparison.json").read_bytes())
    assert len(payload["comparisons"]["A1 minus A0"]["per_seed"]) == 3
    assert payload["comparisons"]["A1 minus A0"]["windows"][1]["window_size"] == 25
    with pytest.raises(FileExistsError, match="overwritten"):
        comparison_report(config, {"A1": 0.1}, models, reports)


def test_ensemble_temperature_cannot_depend_on_development_labels(tmp_path: Path) -> None:
    _, models, _, _ = comparison_fixture(tmp_path)
    cal = tuple(
        read_predictions(models / f"A0-seed{seed}-penalty0" / "calibration.json")
        for seed in (0, 1, 2)
    )
    dev = tuple(
        read_predictions(models / f"A0-seed{seed}-penalty0" / "development.json")
        for seed in (0, 1, 2)
    )
    changed = tuple(tuple(replace(row, label=1 - row.label) for row in seed) for seed in dev)
    assert calibrated_ensemble(cal, dev).fit == calibrated_ensemble(cal, changed).fit
    broken = tuple(tuple(replace(row, raw_probability=None) for row in seed) for seed in cal)
    with pytest.raises(ValueError, match="raw logit"):
        calibrated_ensemble(broken, dev)


def test_comparison_reuses_verified_shared_explicit_baseline(tmp_path: Path) -> None:
    config, models, reports, _ = comparison_fixture(tmp_path, shared=True)
    comparison_report(config, {"C1": 0.1}, models, reports)
    asset = FrozenEnsemble.model_validate_json((reports / "ensemble-C0.json").read_bytes())
    assert all("response-cycle" in seed.directory for seed in asset.seed_runs)


def test_comparison_rejects_smoke_missing_matrix_and_changed_code(tmp_path: Path) -> None:
    config, models, reports, _ = comparison_fixture(tmp_path)
    with pytest.raises(PermissionError):
        comparison_report(
            config.model_copy(update={"stage": "preparation/smoke"}), {"A1": 0.1}, models, reports
        )
    with pytest.raises(ValueError, match="one registered penalty"):
        comparison_report(config, {}, models, reports)
    directory = models / "A1-seed1-penalty0.1"
    manifest = load_run(directory)
    changed = manifest.model_copy(
        update={
            "runtime": tuple(
                (key, "c" * 64 if key == "code_sha256" else value)
                for key, value in manifest.runtime
            )
        }
    )
    (directory / "manifest.json").write_text(changed.model_dump_json())
    with pytest.raises(ValueError, match="unchanged code"):
        comparison_report(config, {"A1": 0.1}, models, reports)
    assert not (reports / "comparison.json").exists()


def test_comparison_rejects_resealed_missing_prediction_row(tmp_path: Path) -> None:
    config, models, reports, _ = comparison_fixture(tmp_path)
    directory = models / "A1-seed2-penalty0.1"
    manifest = load_run(directory)
    path = directory / "development.json"
    payload = json.loads(path.read_bytes())
    path.write_text(json.dumps(payload[:-1]))
    changed = manifest.model_copy(
        update={
            "outputs": tuple(
                file_record(path, directory) if member.path == path.name else member
                for member in manifest.outputs
            )
        }
    )
    (directory / "manifest.json").write_text(changed.model_dump_json())
    with pytest.raises(ValueError, match="frozen evaluation role"):
        comparison_report(config, {"A1": 0.1}, models, reports)


def test_publication_refuses_overwrite_and_cleans_owned_partial_outputs(tmp_path: Path) -> None:
    report = tmp_path / "report"
    with pytest.raises(FileNotFoundError):
        publish_comparison(report, {"ensemble.json": b"{}"})
    assert not (report / "ensemble.json").exists()
    assert not (report / ".comparison.lock").exists()
    assert not tuple(report.glob(".comparison-*"))
