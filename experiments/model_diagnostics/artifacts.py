"""Immutable diagnostic trials bind configuration, source population, and code."""

import os
import shutil
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from pydantic import Field

from clash_sos.domain.canonical_dataset import canonical_json_bytes
from clash_sos.domain.manifests import ManifestModel, Sha256
from experiments.common.artifacts import file_record, verify_files
from experiments.common.contracts import (
    FileRecord,
    OptimizerConfig,
    PopulationIdentity,
    fingerprint,
)
from experiments.common.provenance import code_digest, revision
from experiments.common.session import Session
from experiments.model_diagnostics.model import ModelSpec


class TrialSpec(ManifestModel):
    model: ModelSpec = ModelSpec()
    optimizer: OptimizerConfig = OptimizerConfig()
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    shuffle_seed: int | None = Field(default=None, ge=0)


class TrialInputs(ManifestModel):
    version: Literal["diagnostic-trial:v1"] = "diagnostic-trial:v1"
    spec: TrialSpec
    population: PopulationIdentity
    schema_sha256: Sha256
    protocol_sha256: Sha256
    code_sha256: Sha256
    git_sha: str
    dirty_sha256: Sha256 | None
    lock_sha256: Sha256


class TrialManifest(ManifestModel):
    inputs: TrialInputs
    status: Literal["complete", "calibration_failed", "failed", "time_limited"]
    outputs: tuple[FileRecord, ...]
    elapsed_seconds: float = Field(ge=0, allow_inf_nan=False)
    failure: str | None = None


def bind_inputs(session: Session, spec: TrialSpec, root: Path) -> TrialInputs:
    sha, dirty, lock = revision(root)
    return TrialInputs(
        spec=spec,
        population=session.population,
        schema_sha256=session.schema.fingerprint(),
        protocol_sha256=fingerprint(session.access.protocol),
        code_sha256=code_digest(root),
        git_sha=sha,
        dirty_sha256=dirty,
        lock_sha256=lock,
    )


def load_trial(directory: Path, expected: TrialInputs | None = None) -> TrialManifest:
    manifest = TrialManifest.model_validate_json((directory / "manifest.json").read_bytes())
    if expected is not None and manifest.inputs != expected:
        raise ValueError("trial configuration, code, or population changed")
    verify_files(directory, manifest.outputs)
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if actual != {"manifest.json", *(f.path for f in manifest.outputs)}:
        raise ValueError("trial contains uninventoried members")
    return manifest


def finish_trial(
    directory: Path,
    inputs: TrialInputs,
    status: Literal["complete", "calibration_failed", "failed", "time_limited"],
    elapsed: float,
    failure: str | None = None,
) -> TrialManifest:
    outputs = tuple(file_record(p, directory) for p in sorted(directory.rglob("*")) if p.is_file())
    manifest = TrialManifest(
        inputs=inputs, status=status, outputs=outputs, elapsed_seconds=elapsed, failure=failure
    )
    with (directory / "manifest.json").open("xb") as target:
        target.write(canonical_json_bytes(manifest.model_dump()))
        target.flush()
        os.fsync(target.fileno())
    return load_trial(directory, inputs)


@contextmanager
def trial_stage(destination: Path) -> Generator[Path]:
    """Publish verified diagnostics atomically; never replace an existing trial."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.with_name(destination.name + ".lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    stage: Path | None = None
    try:
        if destination.exists():
            raise FileExistsError(destination)
        stage = Path(tempfile.mkdtemp(prefix=".diagnostic-", dir=destination.parent))
        yield stage
        load_trial(stage)
        stage.rename(destination)
        stage = None
    finally:
        os.close(descriptor)
        lock.unlink()
        if stage is not None:
            shutil.rmtree(stage)
