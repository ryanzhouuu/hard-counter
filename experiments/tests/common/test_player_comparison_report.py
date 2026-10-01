import json
from dataclasses import replace
from pathlib import Path

import pytest
from experiments.common.artifacts import file_record, load_run
from experiments.common.comparison import comparison_report
from experiments.common.execution import job_name
from experiments.common.matrix import DevelopmentResult, screen_jobs, select_penalties
from experiments.common.predictions import read_predictions, write_predictions
from experiments.player_adjustment.diagnostics import matchup_probability_stability
from experiments.tests.common.comparison_fixture import comparison_fixture


def test_player_comparison_publishes_actual_metrics_and_seed_diagnostics(tmp_path: Path) -> None:
    config, models, reports, _ = comparison_fixture(tmp_path, player=True)
    report = comparison_report(config, {"B1": 0.1, "B2": 0.1}, models, reports)
    serialized = json.loads((reports / "comparison.json").read_bytes())
    assert report["player_adjustment"]
    player = serialized["player_adjustment"]
    for identity in ("B0", "B1", "B2"):
        model = player["models"][identity]
        assert model["development"]["metrics"]["row_count"] == 32
        assert model["calibration"]["row_count"] == 32
        assert len(model["seeds"]) == 3
        if identity != "B0":
            for seed in model["seeds"]:
                assert len(seed["diagnostics"]["evaluation_support"]) == 32
                assert seed["diagnostics"]["nuisance_shrinkage"]["branch"] in ("history", "joint")
        actual_cal = read_predictions(reports / f"actual-ensemble-{identity}-calibration.json")
        first_seed = (
            read_predictions(models / f"{identity}-seed0-penalty0.1" / "actual-calibration.json")
            if identity != "B0"
            else None
        )
        if first_seed is not None:
            raw_seeds = [
                read_predictions(
                    models / f"{identity}-seed{seed}-penalty0.1" / "actual-calibration.json"
                )
                for seed in config.seeds
            ]
            assert actual_cal[0].raw_probability == pytest.approx(
                sum(seed[0].raw_probability or 0 for seed in raw_seeds) / len(raw_seeds)
            )
    baseline = read_predictions(reports / "ensemble-B0-development.json")
    matchup = read_predictions(reports / "ensemble-B1-development.json")
    expected = matchup_probability_stability(
        [row.probability for row in baseline], [row.probability for row in matchup]
    )
    comparison = player["comparisons"]["B1 minus B0"]
    assert comparison["matchup_probability_stability"] == expected
    assert comparison["actual_outcome_paired"]["candidate"]["metrics"]["row_count"] == 32


@pytest.mark.parametrize(
    "damage", ["missing_development", "missing_calibration", "missing_report", "orientation"]
)
def test_player_comparison_rejects_invalid_actual_diagnostic_assets(
    tmp_path: Path, damage: str
) -> None:
    config, models, reports, _ = comparison_fixture(tmp_path, player=True)
    directory = models / "B1-seed1-penalty0.1"
    manifest = load_run(directory)
    path = directory / {
        "missing_calibration": "actual-calibration.json",
        "missing_report": "report.json",
    }.get(damage, "actual-development.json")
    if damage.startswith("missing"):
        path.unlink()
    else:
        rows = read_predictions(path)
        write_predictions(path, (replace(rows[0], player_a="another-player"), *rows[1:]))
    outputs = tuple(
        file_record(directory / member.path, directory)
        for member in manifest.outputs
        if (directory / member.path).exists()
    )
    (directory / "manifest.json").write_text(
        manifest.model_copy(update={"outputs": outputs}).model_dump_json()
    )
    with pytest.raises(ValueError, match=r"actual-outcome|orientation"):
        comparison_report(config, {"B1": 0.1, "B2": 0.1}, models, reports)
    assert not (reports / "comparison.json").exists()


def test_actual_outcome_diagnostics_cannot_change_penalty_selection(tmp_path: Path) -> None:
    config, models, _, session = comparison_fixture(tmp_path, player=True)
    results = [
        DevelopmentResult(job, models / job_name(job), "development.json")
        for job in screen_jobs(config)
        if job.penalty > 0
    ]
    before = select_penalties(config, results, session.access)
    for result in results:
        manifest = load_run(result.directory)
        path = result.directory / "actual-development.json"
        write_predictions(
            path,
            tuple(replace(row, probability=1 - row.probability) for row in read_predictions(path)),
        )
        outputs = tuple(
            file_record(path, result.directory) if member.path == path.name else member
            for member in manifest.outputs
        )
        (result.directory / "manifest.json").write_text(
            manifest.model_copy(update={"outputs": outputs}).model_dump_json()
        )
    assert select_penalties(config, results, session.access) == before
