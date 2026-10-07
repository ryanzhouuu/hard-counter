"""Descriptive trial and seed comparisons; never emit a promotion decision."""

import json
from dataclasses import asdict, replace
from pathlib import Path

from pydantic_core import to_jsonable_python

from experiments.common.contracts import fingerprint
from experiments.common.predictions import Prediction, read_predictions
from experiments.common.statistics import paired_comparison, score_predictions
from experiments.model_diagnostics.artifacts import load_trial
from experiments.model_diagnostics.batch import freeze_json


def summarize(directory: Path) -> dict[str, object]:
    state = json.loads((directory / "progress.json").read_bytes())
    if state["status"] != "complete":
        raise ValueError("batch must complete before finalist comparisons")
    definition = json.loads((directory / "definition.json").read_bytes())
    expected = {key: value for key, value in definition["inputs"].items() if key != "spec"}
    table: list[dict[str, object]] = []
    reports: dict[str, object] = {}
    manifests: dict[str, str] = {}
    for name in state["trials"]:
        path = directory / "trials" / name
        manifest = load_trial(path)
        current = manifest.inputs.model_dump(mode="json", exclude={"spec"})
        if current != expected:
            raise ValueError("trial provenance differs from the frozen batch")
        manifests[name] = fingerprint(manifest)
        report = json.loads((path / "report.json").read_bytes())
        metrics = report["metrics"].get("actual-development")
        table.append(
            {
                "name": name,
                "status": manifest.status,
                "model": manifest.inputs.spec.model,
                "seed": manifest.inputs.spec.seed,
                "selected_epoch": report["selected_epoch"],
                "fit_seconds": report["fit_seconds"],
                "actual": metrics,
                "matchup_shared": report["metrics"].get("shared-matchup-development"),
                "matchup_separate": report["metrics"].get("separate-matchup-development"),
                "raw_actual": report["metrics"]["actual-development-raw"],
                "temperatures": report["temperatures"],
                "player_effects": report["player_effects"],
            }
        )
        reports[name] = report
    baseline, *players = state["finalists"]
    comparisons: dict[str, object] = {}
    for player in players:
        for seed in range(3):
            suffix = "" if seed == 0 else f"-seed{seed}"
            candidate, reference = player + suffix, baseline + suffix
            for output in ("actual-development", "shared-matchup-development"):
                first = directory / "trials" / candidate / f"{output}.json"
                second = directory / "trials" / reference / f"{output}.json"
                comparisons[f"{candidate} minus {reference}: {output}"] = (
                    asdict(paired_comparison(read_predictions(first), read_predictions(second)))
                    if first.exists() and second.exists()
                    else {"status": "calibration unavailable"}
                )
    raw = read_predictions(directory / "trials/reference/actual-development-raw.json")
    uniform: tuple[Prediction, ...] = tuple(
        replace(r, logit=0, probability=0.5, raw_probability=0.5) for r in raw
    )
    return {
        "purpose": "exploratory diagnostics; no promotion or equal-skill identification claim",
        "trial_manifests": manifests,
        "table": table,
        "finalists": state["finalists"],
        "constant_half": asdict(score_predictions(uniform)),
        "paired_by_seed": comparisons,
        "trials": reports,
        "uncertainty": "unadjusted shared-player intervals; omit training and day/meta dependence",
        "seed_aggregation": "seeds reported separately; no fitted ensemble or frozen candidate",
    }


def write_summary(directory: Path) -> Path:
    report = to_jsonable_python(summarize(directory))
    freeze_json(directory / "summary.json", report)
    lines = [
        "# Capacity and player diagnostics",
        "",
        "Exploratory development results; no model promotion or equal-skill identification.",
        "",
        "| Trial | Full loss | Matchup loss (shared T) | Accuracy | Epoch |",
        "|---|---:|---:|---:|---:|",
    ]
    for entry in report["table"]:
        full, match = entry["actual"], entry["matchup_shared"]
        loss = f"{full['metrics']['log_loss']:.6f}" if full else "unavailable"
        match_loss = f"{match['metrics']['log_loss']:.6f}" if match else "unavailable"
        accuracy = f"{full['accuracy_half_ties']:.2%}" if full else "unavailable"
        lines.append(
            f"| {entry['name']} | {loss} | {match_loss} | {accuracy} | {entry['selected_epoch']} |"
        )
    lines.extend(
        [
            "",
            "Finalists: " + ", ".join(report["finalists"]),
            "",
            "Paired intervals, daily/history slices, learning curves, calibration, and "
            "shrinkage diagnostics are in summary.json. Seeds are not independent samples "
            "of the battle population. Screening introduces selection bias; future confirmation "
            "remains necessary. Shared-temperature matchup scores remove the player term "
            "before applying the full-score temperature. Separately calibrated outputs are "
            "retained as a sensitivity analysis.",
        ]
    )
    destination = directory / "summary.md"
    content = "\n".join(lines) + "\n"
    if destination.exists() and destination.read_text() != content:
        raise ValueError("existing summary differs from verified trials")
    destination.write_text(content)
    return destination
