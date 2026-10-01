"""Verified player-aware outputs remain diagnostic alongside primary matchup comparisons."""

from __future__ import annotations

import json
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, cast

from pydantic_core import to_jsonable_python

from clash_sos.domain.canonical_dataset import canonical_json_bytes
from experiments.common.contracts import RunManifest, Variant
from experiments.common.ensembles import EnsemblePredictions, calibrated_ensemble
from experiments.common.predictions import Prediction, pair_predictions, read_predictions
from experiments.common.statistics import paired_comparison, score_predictions
from experiments.player_adjustment.diagnostics import matchup_probability_stability

if TYPE_CHECKING:
    from experiments.common.comparison_inputs import ComparisonRun


def load_actual_outputs(
    directory: Path,
    manifest: RunManifest,
    variant: Variant,
    calibration: tuple[Prediction, ...],
    development: tuple[Prediction, ...],
) -> tuple[tuple[Prediction, ...], tuple[Prediction, ...], dict[str, object] | None]:
    """Unadjusted models share outputs; nuisance models require exact inventoried role parity."""
    if variant.nuisance == "none":
        return calibration, development, None
    required = {"actual-calibration.json", "actual-development.json", "report.json"}
    if not required <= {member.path for member in manifest.outputs}:
        raise ValueError(
            "player comparisons require inventoried actual-outcome predictions and diagnostics"
        )
    actual_cal, actual_dev = (
        read_predictions(directory / name)
        for name in ("actual-calibration.json", "actual-development.json")
    )
    pair_predictions(actual_cal, calibration)
    pair_predictions(actual_dev, development)
    raw: object = json.loads((directory / "report.json").read_bytes())
    if not isinstance(raw, dict):
        raise ValueError("player comparison requires published player diagnostics")
    diagnostic = cast(dict[str, object], raw).get("player_diagnostics")
    if not isinstance(diagnostic, dict):
        raise ValueError("player comparison requires published player diagnostics")
    diagnostic = cast(dict[str, object], diagnostic)
    return actual_cal, actual_dev, diagnostic


def player_comparison_report(
    runs: dict[str, tuple[ComparisonRun, ...]],
    matchup: dict[str, EnsemblePredictions],
    payloads: dict[str, bytes],
) -> dict[str, object]:
    """Calibrate actual-output ensembles on calibration only; never feed them to selection."""
    actual: dict[str, EnsemblePredictions] = {}
    models: dict[str, object] = {}
    for identity, seeds in runs.items():
        ensemble = (
            matchup[identity]
            if seeds[0].checkpoint.metadata.variant.nuisance == "none"
            else calibrated_ensemble(
                tuple(run.actual_calibration for run in seeds),
                tuple(run.actual_development for run in seeds),
            )
        )
        actual[identity] = ensemble
        for role in ("calibration", "development"):
            payloads[f"actual-ensemble-{identity}-{role}.json"] = canonical_json_bytes(
                to_jsonable_python(getattr(ensemble, role))
            )
        models[identity] = {
            "calibration": asdict(ensemble.fit),
            "development": score_predictions(ensemble.development),
            "seeds": [
                {
                    "seed": run.asset.seed,
                    "calibration": score_predictions(run.actual_calibration),
                    "development": score_predictions(run.actual_development),
                    "diagnostics": run.player_diagnostics,
                }
                for run in seeds
            ],
        }
    comparisons: dict[str, object] = {}
    for comparator, candidate in combinations(runs, 2):
        aligned = pair_predictions(matchup[candidate].development, matchup[comparator].development)
        comparisons[f"{candidate} minus {comparator}"] = {
            "actual_outcome_paired": paired_comparison(
                actual[candidate].development, actual[comparator].development
            ),
            "matchup_probability_stability": matchup_probability_stability(
                tuple(second.probability for _, second in aligned),
                tuple(first.probability for first, _ in aligned),
            ),
        }
    return {
        "interpretation": "actual-outcome outputs are diagnostic; selection uses matchup-only loss",
        "models": models,
        "comparisons": comparisons,
    }
