"""Retain calibration-only evidence even when the guarded temperature fit fails."""

from dataclasses import asdict
from math import exp, log
from pathlib import Path
from typing import Literal

import numpy as np

from clash_sos.domain.canonical_dataset import canonical_json_bytes
from experiments.common.calibration import TemperatureFit, fit_temperature
from experiments.common.checkpoints import predictions
from experiments.common.data_access import ResearchRow
from experiments.common.predictions import write_predictions


def fit_recorded_temperature(
    stage: Path,
    rows: tuple[ResearchRow, ...],
    logits: np.ndarray,
    *,
    output: Literal["matchup", "actual"],
) -> TemperatureFit:
    """Save raw evidence separately from calibrated assets; propagate every fit rejection."""
    raw = predictions(rows, logits, 1)
    write_predictions(stage / f"{output}-calibration-raw.json", raw)
    labels = tuple(r.label for r in rows)
    values = tuple(float(z) for z in logits)
    signed = np.asarray([(2 * y - 1) * z for y, z in zip(labels, values, strict=True)])
    bounds = (-4.0, 4.0)
    document: dict[str, object] = {
        "purpose": "calibration diagnostics; raw scores are not calibrated assets",
        "row_count": len(rows),
        "positive_labels": sum(labels),
        "log_temperature_bounds": bounds,
        "raw_loss": float(np.mean(np.logaddexp(0, -signed))),
        "uniform_prediction_loss": log(2),
        "logit_range": (min(values), max(values)),
        "mean_signed_logit": float(np.mean(signed)),
        "inverse_temperature_derivative_at_zero": -float(np.mean(signed)) / 2,
        "loss_curve": tuple(
            {
                "log_temperature": point,
                "temperature": exp(point),
                "loss": float(np.mean(np.logaddexp(0, -signed / exp(point)))),
            }
            for point in np.linspace(*bounds, 9).tolist()
        ),
    }
    path = stage / f"{output}-calibration-diagnostics.json"
    try:
        fitted = fit_temperature(values, labels, log_temperature_bounds=bounds)
    except ValueError as error:
        document.update(status="failed", reason=str(error))
        path.write_bytes(canonical_json_bytes(document))
        raise
    document.update(status="complete", fit=asdict(fitted))
    path.write_bytes(canonical_json_bytes(document))
    return fitted
