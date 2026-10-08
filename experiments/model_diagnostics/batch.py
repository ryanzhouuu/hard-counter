"""Resume a declared diagnostic search without consuming reporting populations."""

import json
import os
from collections.abc import Callable
from dataclasses import asdict
from math import isfinite
from pathlib import Path
from time import monotonic

from pydantic_core import to_jsonable_python

from clash_sos.domain.canonical_dataset import canonical_json_bytes
from experiments.common.contracts import OptimizerConfig
from experiments.common.session import Session
from experiments.model_diagnostics.artifacts import bind_inputs, load_trial
from experiments.model_diagnostics.schedule import (
    PLAYER_PENALTIES,
    Job,
    baseline_jobs,
    branch_jobs,
    finalist_jobs,
    player_jobs,
    select_job,
)
from experiments.model_diagnostics.trial import run_trial

DEFAULT_OPTIMIZER = OptimizerConfig(max_epochs=80, patience=8)


def freeze_json(path: Path, value: object) -> None:
    payload = canonical_json_bytes(to_jsonable_python(value))
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"frozen diagnostic file changed: {path.name}")
    else:
        with path.open("xb") as target:
            target.write(payload)


def run_batch(
    directory: Path,
    session: Session,
    root: Path,
    *,
    optimizer: OptimizerConfig = DEFAULT_OPTIMIZER,
    budget_seconds: float = 8 * 60 * 60,
    progress: Callable[[str], None] = print,
) -> dict[str, object]:
    if not 0 < budget_seconds <= 8 * 60 * 60:
        raise ValueError("diagnostic batch budget must be positive and at most eight hours")
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".batch.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        return _execute(directory, session, root, optimizer, budget_seconds, progress)
    finally:
        os.close(descriptor)
        lock.unlink()


def _execute(
    directory: Path,
    session: Session,
    root: Path,
    optimizer: OptimizerConfig,
    budget: float,
    progress: Callable[[str], None],
) -> dict[str, object]:
    started = monotonic()
    backbones = baseline_jobs(optimizer)
    bound = bind_inputs(session, backbones[0].spec, root)
    freeze_json(
        directory / "definition.json",
        {
            "inputs": bound,
            "backbones": tuple(asdict(j) for j in backbones),
            "player_penalties": PLAYER_PENALTIES,
            "extra_seeds": (1, 2),
            "shuffle_seed": 9127,
            "budget_seconds": budget,
            "maximum_fit_count": 26,
            "selection": "full calibrated development log loss; fixed-order ties; exploratory only",
        },
    )
    state_path = directory / "progress.json"
    used = (
        float(json.loads(state_path.read_text())["elapsed_seconds"]) if state_path.exists() else 0
    )
    if not isfinite(used) or used < 0:
        raise ValueError("invalid accumulated batch runtime")
    result: dict[str, object] = {"status": "running", "trials": {}, "elapsed_seconds": used}
    records: dict[str, object] = {}

    def save(status: str) -> dict[str, object]:
        result.update(status=status, trials=records, elapsed_seconds=used + monotonic() - started)
        temporary = state_path.with_suffix(".tmp")
        temporary.write_bytes(canonical_json_bytes(to_jsonable_python(result)))
        temporary.replace(state_path)
        return result

    def execute(job: Job) -> float | None:
        if bind_inputs(session, backbones[0].spec, root) != bound:
            raise ValueError("batch source or population changed during execution")
        destination = directory / "trials" / job.name
        if not destination.exists() and budget - used - (monotonic() - started) < (
            job.spec.optimizer.time_limit_seconds
        ):
            raise TimeoutError("remaining batch budget cannot accommodate the next bounded fit")
        progress(f"Diagnostic trial: {job.name}")
        manifest = run_trial(destination, session, job.spec, root)
        records[job.name] = {"status": manifest.status, "elapsed_seconds": manifest.elapsed_seconds}
        save("running")
        if manifest.status in ("failed", "time_limited"):
            raise RuntimeError(f"{job.name}: {manifest.failure}")
        report = json.loads((destination / "report.json").read_text())
        value = report["selection_loss"]
        return None if value is None else float(value)

    try:
        losses = {job.name: execute(job) for job in backbones}
        baseline = select_job(backbones, losses)
        freeze_json(
            directory / "backbone-selection.json", {"job": asdict(baseline), "losses": losses}
        )
        negative = Job(
            "shuffled-reference", backbones[0].spec.model_copy(update={"shuffle_seed": 9127})
        )
        execute(negative)
        players = player_jobs(baseline)
        losses = {job.name: execute(job) for job in players}
        selected = [baseline]
        for branch in ("history", "joint"):
            candidates = branch_jobs(players, branch)
            selected.append(select_job(candidates, {j.name: losses[j.name] for j in candidates}))
        freeze_json(
            directory / "player-selection.json",
            {
                "jobs": tuple(asdict(j) for j in selected),
                "losses": losses,
            },
        )
        result["finalists"] = [j.name for j in selected]
        for job in finalist_jobs(*selected):
            execute(job)
        for job in (*backbones, negative, *players, *finalist_jobs(*selected)):
            load_trial(directory / "trials" / job.name, bind_inputs(session, job.spec, root))
        return save("complete")
    except TimeoutError as error:
        result["reason"] = str(error)
        return save("budget_exhausted")
    except (ValueError, RuntimeError, OSError) as error:
        result["reason"] = str(error)
        return save("incomplete")
    except KeyboardInterrupt:
        save("interrupted")
        raise
