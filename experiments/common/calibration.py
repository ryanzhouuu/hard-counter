"""Intercept-free scalar calibration, restricted to supplied calibration rows."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import exp, isfinite, log, log1p
from typing import cast

from scipy.optimize import minimize_scalar

from experiments.common.predictions import Prediction


@dataclass(frozen=True)
class TemperatureFit:
    temperature: float
    row_count: int
    raw_loss: float
    calibrated_loss: float
    log_temperature_bounds: tuple[float, float]


def sigmoid(logit: float, temperature: float = 1.0) -> float:
    if not isfinite(logit) or not isfinite(temperature) or temperature <= 0:
        raise ValueError("calibration requires finite logits and positive finite temperature")
    scaled = logit / temperature
    if scaled >= 0:
        return 1 / (1 + exp(-scaled))
    exponential = exp(scaled)
    return exponential / (1 + exponential)


def _loss(logits: Sequence[float], labels: Sequence[int], temperature: float) -> float:
    total = 0.0
    for logit, label in zip(logits, labels, strict=True):
        signed = (1 - 2 * label) * logit / temperature
        total += max(signed, 0) + log1p(exp(-abs(signed)))
    return total / len(labels)


def fit_temperature(
    logits: Sequence[float],
    labels: Sequence[int],
    *,
    log_temperature_bounds: tuple[float, float] = (-4.0, 4.0),
) -> TemperatureFit:
    """Bounds restrict temperature to exp(-4)..exp(4) unless explicitly overridden."""
    if not logits or len(logits) != len(labels):
        raise ValueError("calibration requires nonempty aligned logits and labels")
    if any(not isfinite(value) for value in logits) or any(label not in (0, 1) for label in labels):
        raise ValueError("calibration requires finite logits and binary labels")
    lower, upper = log_temperature_bounds
    if not isfinite(lower) or not isfinite(upper) or not -700 < lower < upper < 700:
        raise ValueError("invalid log-temperature bounds")
    if not any(value != 0 for value in logits):
        raise ValueError("temperature is unidentified for all-zero logits")

    def objective(log_temperature: float) -> float:
        return _loss(logits, labels, exp(log_temperature))

    result = minimize_scalar(objective, bounds=(lower, upper), method="bounded")
    optimum = cast(float, result.x)
    if not result.success or not isfinite(optimum) or not isfinite(cast(float, result.fun)):
        raise ValueError("temperature optimizer failed")
    if min(optimum - lower, upper - optimum) < 1e-3:
        raise ValueError("temperature optimizer reached a boundary solution")
    temperature = exp(optimum)
    return TemperatureFit(
        temperature, len(labels), _loss(logits, labels, 1), objective(optimum), (lower, upper)
    )


def calibrate_predictions(rows: Sequence[Prediction], temperature: float) -> tuple[Prediction, ...]:
    return tuple(
        replace(
            row, probability=sigmoid(row.logit, temperature), raw_probability=sigmoid(row.logit)
        )
        for row in rows
    )


def ensemble_logits(probabilities: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """Average seed probabilities before applying one ensemble temperature."""
    if not probabilities or not probabilities[0]:
        raise ValueError("ensemble requires nonempty seed probabilities")
    if any(len(seed) != len(probabilities[0]) for seed in probabilities):
        raise ValueError("ensemble seed rows must align")
    output: list[float] = []
    for values in zip(*probabilities, strict=True):
        if any(not isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError("ensemble probabilities must be finite and in [0, 1]")
        mean = sum(values) / len(values)
        if not 0 < mean < 1:
            raise ValueError("ensemble mean must be strictly inside (0, 1) for finite logits")
        output.append(log(mean) - log1p(-mean))
    return tuple(output)
