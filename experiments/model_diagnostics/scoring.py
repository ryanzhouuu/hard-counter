"""Compare full-score calibration with skill-removed outputs without refitting skill."""

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC
from pathlib import Path
from typing import Literal, cast

import numpy as np

from experiments.common.calibration import TemperatureFit
from experiments.common.calibration_diagnostics import fit_recorded_temperature
from experiments.common.checkpoints import predictions
from experiments.common.data_access import ResearchRow
from experiments.common.predictions import Prediction, write_predictions
from experiments.common.statistics import score_predictions


@dataclass(frozen=True)
class Evaluation:
    selection_loss: float | None
    actual_temperature: float | None
    matchup_temperature: float | None
    predictions: dict[str, tuple[Prediction, ...]]
    report: dict[str, object]


def evaluate(
    directory: Path,
    calibration: tuple[ResearchRow, ...],
    development: tuple[ResearchRow, ...],
    actual: tuple[np.ndarray, np.ndarray],
    matchup: tuple[np.ndarray, np.ndarray],
    training: tuple[ResearchRow, ...],
) -> Evaluation:
    """Rejected calibration preserves raw diagnostics and excludes full-score selection."""
    if max(r.key[0] for r in calibration) >= min(r.key[0] for r in development):
        raise ValueError("calibration must strictly precede development")
    if {r.event_key for r in calibration} & {r.event_key for r in development}:
        raise ValueError("calibration and development events must be disjoint")
    fits: dict[str, TemperatureFit | None] = {}
    errors: dict[str, str] = {}
    output: dict[str, tuple[Prediction, ...]] = {}
    for name, values in (("actual", actual), ("matchup", matchup)):
        for rows, logits in zip((calibration, development), values, strict=True):
            if logits.shape != (len(rows),) or not np.isfinite(logits).all():
                raise ValueError("finite logits must align with role rows")
        try:
            fits[name] = fit_recorded_temperature(
                directory, calibration, values[0], output=cast(Literal["actual", "matchup"], name)
            )
        except ValueError as error:
            fits[name] = None
            errors[name] = str(error)
        for role, rows, logits in zip(
            ("calibration", "development"), (calibration, development), values, strict=True
        ):
            output[f"{name}-{role}-raw"] = predictions(rows, logits, 1)
    full, separate = fits["actual"], fits["matchup"]
    for role, rows, full_z, match_z in zip(
        ("calibration", "development"), (calibration, development), actual, matchup, strict=True
    ):
        if full is not None:
            output[f"actual-{role}"] = predictions(rows, full_z, full.temperature)
            output[f"shared-matchup-{role}"] = predictions(rows, match_z, full.temperature)
        if separate is not None:
            output[f"separate-matchup-{role}"] = predictions(rows, match_z, separate.temperature)
    for name, rows in output.items():
        write_predictions(directory / f"{name}.json", rows)
    metrics = {
        name: asdict(score_predictions(rows))
        for name, rows in output.items()
        if "development" in name
    }
    support = Counter(p for r in training for p in (r.player_a, r.player_b))
    slices: dict[str, object] = {}
    for name, rows in output.items():
        if "development" not in name:
            continue
        groups: dict[str, list[Prediction]] = {}
        for row in rows:
            keys = (
                f"day:{row.timestamp.astimezone(UTC).date()}",
                f"seen_players:{sum(support[p] > 0 for p in (row.player_a, row.player_b))}",
                f"both_history_10:{min(support[row.player_a], support[row.player_b]) >= 10}",
            )
            for key in keys:
                groups.setdefault(key, []).append(row)
        slices[name] = {key: asdict(score_predictions(group)) for key, group in groups.items()}
    loss = (
        score_predictions(output["actual-development"]).metrics.log_loss
        if full is not None
        else None
    )
    return Evaluation(
        loss,
        full.temperature if full else None,
        separate.temperature if separate else None,
        output,
        {
            "selection_loss": loss,
            "selection_target": "calibrated full outcome loss; exploratory screening only",
            "calibration_errors": errors,
            "temperatures": {name: asdict(fit) if fit else None for name, fit in fits.items()},
            "metrics": metrics,
            "slices": slices,
            "interpretation": "additive skill removal; equal skill is not identified",
            "slice_support": "counts are descriptive; ten prior games is not a sufficiency claim",
        },
    )
