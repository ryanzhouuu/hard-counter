"""Registered screening schedules and development-only penalty selection."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Literal

from clash_sos.domain.model_artifact import log_loss
from experiments.common.artifacts import load_run
from experiments.common.contracts import StudyConfig, Variant, fingerprint
from experiments.common.data_access import RoleAccess
from experiments.common.predictions import read_predictions

Phase = Literal["smoke", "screen", "confirmation", "reference"]


@dataclass(frozen=True)
class MatrixJob:
    variant_id: str
    seed: int
    penalty: float
    phase: Phase
    shared: bool = False

    def __post_init__(self) -> None:
        if not self.variant_id or self.seed < 0 or not isfinite(self.penalty) or self.penalty < 0:
            raise ValueError(
                "matrix settings require a variant, nonnegative seed, and finite penalty"
            )
        if self.phase not in ("smoke", "screen", "confirmation", "reference"):
            raise ValueError("unregistered matrix phase")
        object.__setattr__(self, "penalty", float(self.penalty))


def _baseline(variant: Variant) -> bool:
    return (
        variant.architecture == "explicit"
        and variant.nuisance == "none"
        and not (variant.feature_groups)
    )


def _shared(config: StudyConfig) -> bool:
    return config.study_id != "response-cycle"


def _require_seed_zero(config: StudyConfig) -> None:
    if 0 not in config.seeds:
        raise ValueError("registered screening requires seed zero")
    if len(set(config.penalties)) != len(config.penalties):
        raise ValueError("registered penalty grid must be unique")


def smoke_jobs(config: StudyConfig) -> tuple[MatrixJob, ...]:
    """Capped correctness fits retain every enabled branch and never rank them."""
    _require_seed_zero(config)
    return tuple(
        MatrixJob(
            v.variant_id,
            0,
            0 if _baseline(v) or v.architecture == "attention" else config.penalties[0],
            "smoke",
        )
        for v in config.variants
        if v.enabled
    )


def screen_jobs(config: StudyConfig) -> tuple[MatrixJob, ...]:
    _require_seed_zero(config)
    jobs: list[MatrixJob] = []
    for variant in config.variants:
        if not variant.enabled or variant.architecture == "attention":
            continue
        if _baseline(variant):
            jobs.append(MatrixJob(variant.variant_id, 0, 0, "screen", _shared(config)))
        else:
            jobs.extend(
                MatrixJob(variant.variant_id, 0, penalty, "screen") for penalty in config.penalties
            )
    return tuple(jobs)


def confirmation_jobs(
    config: StudyConfig, selected_penalties: Mapping[str, float]
) -> tuple[MatrixJob, ...]:
    _require_seed_zero(config)
    challengers = {
        v.variant_id
        for v in config.variants
        if v.enabled and v.architecture == "explicit" and not _baseline(v)
    }
    if set(selected_penalties) != challengers or any(
        value not in config.penalties for value in selected_penalties.values()
    ):
        raise ValueError("confirmation requires one registered penalty per enabled challenger")
    jobs: list[MatrixJob] = []
    for variant in config.variants:
        if not variant.enabled:
            continue
        if variant.architecture == "attention":
            jobs.extend(
                MatrixJob(variant.variant_id, seed, 0, "reference") for seed in config.seeds
            )
        else:
            for seed in config.seeds:
                if seed != 0:
                    jobs.append(
                        MatrixJob(
                            variant.variant_id,
                            seed,
                            0 if _baseline(variant) else selected_penalties[variant.variant_id],
                            "confirmation",
                            _baseline(variant) and _shared(config),
                        )
                    )
    return tuple(jobs)


@dataclass(frozen=True)
class DevelopmentResult:
    job: MatrixJob
    directory: Path
    predictions_path: str


def development_loss(config: StudyConfig, result: DevelopmentResult, access: RoleAccess) -> float:
    if config.stage != "development-frozen":
        raise ValueError("controlled comparisons require development-frozen inputs")
    if result.job not in screen_jobs(config):
        raise ValueError("comparison settings were not registered for screening")
    run = load_run(result.directory)
    if run.status != "complete" or not run.eligible_for_comparison:
        raise ValueError("comparison requires a complete eligible run")
    if run.config_sha256 != fingerprint(config) or run.population != config.population:
        raise ValueError("comparison configuration or population hash mismatch")
    inputs = {member.path: member for member in run.inputs}
    if any(inputs.get(member.path) != member for member in run.population.snapshot_files):
        raise ValueError("comparison source hashes do not match the population inventory")
    runtime = dict(run.runtime)
    if len(runtime) != len(run.runtime) or any(
        runtime.get(name) != value
        for name, value in (
            ("variant_id", result.job.variant_id),
            ("seed", str(result.job.seed)),
            ("penalty", str(result.job.penalty)),
            ("phase", result.job.phase),
        )
    ):
        raise ValueError("run settings do not match the registered job")
    if result.predictions_path not in {member.path for member in run.outputs}:
        raise ValueError("development predictions must be inventoried")
    predictions = read_predictions(result.directory / result.predictions_path)
    expected = access.read("development", "compare")
    actual = {p.row_key: (p.event_key, p.player_a, p.player_b, p.label) for p in predictions}
    wanted = {r.key: (r.event_key, r.player_a, r.player_b, r.label) for r in expected}
    if not expected or actual != wanted:
        raise ValueError("development predictions require identical oriented rows")
    return log_loss([p.label for p in predictions], [p.probability for p in predictions])


def select_penalties(
    config: StudyConfig, results: Sequence[DevelopmentResult], access: RoleAccess
) -> dict[str, float]:
    if config.stage != "development-frozen":
        raise ValueError("smoke and reporting outcomes cannot select penalties")
    expected = {job for job in screen_jobs(config) if job.penalty > 0}
    indexed = {result.job: result for result in results}
    if len(indexed) != len(results) or set(indexed) != expected:
        raise ValueError("selection requires the complete registered challenger grid exactly once")
    losses = {job: development_loss(config, result, access) for job, result in indexed.items()}
    selected: dict[str, float] = {}
    for variant in config.variants:
        jobs = [job for job in expected if job.variant_id == variant.variant_id]
        if jobs:
            winner = min(jobs, key=lambda job: (losses[job], config.penalties.index(job.penalty)))
            selected[variant.variant_id] = winner.penalty
    return selected
