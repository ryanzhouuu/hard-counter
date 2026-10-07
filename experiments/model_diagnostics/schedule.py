"""Bounded exploratory screening rules, separate from frozen candidate selection."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal

from experiments.common.contracts import OptimizerConfig
from experiments.model_diagnostics.artifacts import TrialSpec
from experiments.model_diagnostics.model import ModelSpec

PLAYER_PENALTIES = (0.001, 0.01, 0.1)


@dataclass(frozen=True)
class Job:
    name: str
    spec: TrialSpec


def baseline_jobs(optimizer: OptimizerConfig) -> tuple[Job, ...]:
    reference = TrialSpec(
        optimizer=optimizer.model_copy(
            update={
                "max_epochs": 20,
                "patience": 3,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
            }
        )
    )
    jobs = [Job("reference", reference)]
    for rate in (0.001, 0.003):
        for strength in (0.01, 0.1):
            jobs.append(
                Job(
                    f"main-lr{rate:g}-l2{strength:g}",
                    TrialSpec(
                        model=ModelSpec(capacity="main", card_l2=strength),
                        optimizer=optimizer.model_copy(update={"learning_rate": rate}),
                    ),
                )
            )
        for strength in (0.001, 0.01, 0.1, 1.0):
            jobs.append(
                Job(
                    f"pairs-lr{rate:g}-l2{strength:g}",
                    TrialSpec(
                        model=ModelSpec(card_l2=0.01, pair_l2=strength),
                        optimizer=optimizer.model_copy(update={"learning_rate": rate}),
                    ),
                )
            )
    return tuple(jobs)


def player_jobs(backbone: Job) -> tuple[Job, ...]:
    jobs: list[Job] = []
    for branch in ("history", "joint"):
        for strength in PLAYER_PENALTIES:
            payload = backbone.spec.model.model_dump()
            payload.update(nuisance=branch, player_l2=strength)
            jobs.append(
                Job(
                    f"{branch}-l2{strength:g}",
                    backbone.spec.model_copy(update={"model": ModelSpec.model_validate(payload)}),
                )
            )
    return tuple(jobs)


def select_job(jobs: tuple[Job, ...], losses: dict[str, float | None]) -> Job:
    """Use full calibrated outcome loss; fixed job order breaks exact ties."""
    if set(losses) != {j.name for j in jobs}:
        raise ValueError("selection requires every registered job exactly once")
    eligible = [j for j in jobs if losses[j.name] is not None]
    if any(value is not None and (not isfinite(value) or value < 0) for value in losses.values()):
        raise ValueError("selection losses must be finite nonnegative values or unavailable")
    if not eligible:
        raise ValueError("no model has an eligible calibrated full-score loss")
    scores = {name: loss for name, loss in losses.items() if loss is not None}
    return min(eligible, key=lambda job: (scores[job.name], jobs.index(job)))


def finalist_jobs(baseline: Job, history: Job, joint: Job) -> tuple[Job, ...]:
    return tuple(
        Job(f"{job.name}-seed{seed}", job.spec.model_copy(update={"seed": seed}))
        for job in (baseline, history, joint)
        for seed in (1, 2)
    )


def branch_jobs(jobs: tuple[Job, ...], branch: Literal["history", "joint"]) -> tuple[Job, ...]:
    return tuple(j for j in jobs if j.spec.model.nuisance == branch)
