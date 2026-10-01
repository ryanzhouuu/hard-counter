"""Verified seed assets and calibration-only probability ensembles."""

import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from clash_sos.domain.canonical_dataset import canonical_json_bytes
from clash_sos.domain.manifests import ManifestModel, Sha256
from experiments.common.calibration import TemperatureFit, ensemble_logits, fit_temperature, sigmoid
from experiments.common.contracts import PopulationIdentity
from experiments.common.predictions import Prediction, pair_predictions


class SeedRun(ManifestModel):
    seed: int = Field(ge=0)
    directory: str
    manifest_sha256: Sha256
    checkpoint_sha256: Sha256
    calibration_sha256: Sha256
    development_sha256: Sha256
    run_config_sha256: Sha256


class FrozenEnsemble(ManifestModel):
    version: Literal["research-ensemble:v1"] = "research-ensemble:v1"
    variant_id: str
    config_sha256: Sha256
    code_sha256: Sha256
    git_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    lock_sha256: Sha256
    population: PopulationIdentity
    temperature: float = Field(gt=0, allow_inf_nan=False)
    calibration_rows_sha256: Sha256
    calibration_row_count: int = Field(gt=0)
    seed_runs: tuple[SeedRun, ...]

    @model_validator(mode="after")
    def seed_inventory(self) -> "FrozenEnsemble":
        if not self.seed_runs or len({run.seed for run in self.seed_runs}) != len(self.seed_runs):
            raise ValueError("frozen ensemble requires unique nonempty seeds")
        if any(not Path(run.directory).is_absolute() for run in self.seed_runs):
            raise ValueError("frozen seed runs require absolute directories")
        return self


@dataclass(frozen=True)
class EnsemblePredictions:
    calibration: tuple[Prediction, ...]
    development: tuple[Prediction, ...]
    fit: TemperatureFit


def _averaged(rows: Sequence[Sequence[Prediction]]) -> tuple[Prediction, ...]:
    if not rows or not rows[0]:
        raise ValueError("ensemble requires nonempty seed rows")
    reference = tuple(pair[0] for pair in pair_predictions(rows[0], rows[0]))
    aligned = tuple(tuple(pair[0] for pair in pair_predictions(seed, reference)) for seed in rows)
    raw: list[tuple[float, ...]] = []
    for seed in aligned:
        probabilities: list[float] = []
        for row in seed:
            if row.raw_probability is None or abs(row.raw_probability - sigmoid(row.logit)) > 1e-12:
                raise ValueError("ensemble seeds require verified raw logit probabilities")
            probabilities.append(row.raw_probability)
        raw.append(tuple(probabilities))
    logits = ensemble_logits(raw)
    return tuple(
        replace(row, logit=logit, probability=sigmoid(logit), raw_probability=sigmoid(logit))
        for row, logit in zip(reference, logits, strict=True)
    )


def calibrated_ensemble(
    calibration: Sequence[Sequence[Prediction]],
    development: Sequence[Sequence[Prediction]],
) -> EnsemblePredictions:
    if len(calibration) != len(development):
        raise ValueError("calibration and development must have identical seed counts")
    cal, dev = _averaged(calibration), _averaged(development)
    if max(row.timestamp for row in cal) >= min(row.timestamp for row in dev):
        raise ValueError("ensemble calibration must strictly precede development timestamps")
    if {row.event_key for row in cal} & {row.event_key for row in dev}:
        raise ValueError("calibration and development event populations must be disjoint")
    fit = fit_temperature(tuple(row.logit for row in cal), tuple(row.label for row in cal))
    return EnsemblePredictions(
        tuple(replace(row, probability=sigmoid(row.logit, fit.temperature)) for row in cal),
        tuple(replace(row, probability=sigmoid(row.logit, fit.temperature)) for row in dev),
        fit,
    )


def publish_comparison(report_root: Path, payloads: dict[str, bytes]) -> None:
    """Publish the completion report last; failures remove only owned newly linked files."""
    report_root.mkdir(parents=True, exist_ok=True)
    lock = report_root / ".comparison.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    stage = Path(tempfile.mkdtemp(prefix=".comparison-", dir=report_root))
    published: list[Path] = []
    try:
        if any((report_root / name).exists() for name in payloads):
            raise FileExistsError("completed comparison outputs cannot be overwritten")
        for name, payload in payloads.items():
            if Path(name).name != name:
                raise ValueError("comparison member names must be local filenames")
            (stage / name).write_bytes(payload)
        ordered = sorted(name for name in payloads if name != "comparison.json")
        for name in (*ordered, "comparison.json"):
            os.link(stage / name, report_root / name)
            published.append(report_root / name)
    except BaseException:
        for path in published:
            path.unlink()
        raise
    finally:
        shutil.rmtree(stage)
        os.close(descriptor)
        lock.unlink()


def ensemble_bytes(asset: FrozenEnsemble) -> bytes:
    return canonical_json_bytes(asset.model_dump(mode="json")) + b"\n"
