from pathlib import Path

from clash_sos.domain.canonical_dataset import canonical_json_bytes
from experiments.common.artifacts import load_run
from experiments.common.contracts import StudyConfig, fingerprint
from experiments.common.matrix import MatrixJob
from experiments.common.provenance import code_digest, revision
from experiments.common.run import run_variant
from experiments.common.session import Session, attention_session


def job_name(job: MatrixJob) -> str:
    return f"{job.variant_id}-seed{job.seed}-penalty{job.penalty:g}"


def execute_jobs(
    config: StudyConfig,
    session: Session,
    jobs: tuple[MatrixJob, ...],
    model_root: Path,
    report_root: Path,
    root: Path,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    calibration = session.access.protocol.calibration
    assert calibration is not None
    for job in jobs:
        variant = next(v for v in config.variants if v.variant_id == job.variant_id)
        selected = (
            attention_session(
                session,
                report_root / "inputs",
                config.smoke_row_cap
                if config.stage == "preparation/smoke"
                else sum(
                    s.row_count
                    for s in (session.access.protocol.refit, session.access.protocol.development)
                )
                + calibration.row_count,
            )
            if variant.architecture == "attention"
            else session
        )
        destination = model_root / job_name(job)
        if job.shared and config.stage == "development-frozen":
            destination = (
                model_root.parent.parent
                / "response-cycle"
                / model_root.name
                / f"A0-seed{job.seed}-penalty0"
            )
            shared = load_run(destination)
            if (
                shared.population != session.population
                or shared.config.optimizer != config.optimizer
                or shared.status != "complete"
                or not shared.eligible_for_comparison
            ):
                raise ValueError("shared baseline inputs must match; run response-cycle first")
            manifest = shared
        elif destination.exists():
            manifest = load_run(destination)
            git_sha, dirty, _ = revision(root)
            if (
                manifest.config_sha256 != fingerprint(config)
                or manifest.population != session.population
                or manifest.git_sha != git_sha
                or manifest.dirty_sha256 != dirty
                or manifest.status != "complete"
                or dict(manifest.runtime).get("code_sha256") != code_digest(root)
            ):
                raise ValueError(
                    "resume requires matching configuration, revision, inputs, and verified outputs"
                )
        else:
            manifest = run_variant(
                root,
                destination,
                config,
                variant,
                selected.access,
                selected.population,
                selected.catalog,
                selected.schema,
                seed=job.seed,
                penalty=job.penalty,
                phase=job.phase,
            )
        report_root.mkdir(parents=True, exist_ok=True)
        summary: dict[str, object] = {
            "variant": job.variant_id,
            "seed": job.seed,
            "penalty": job.penalty,
            "status": manifest.status,
            "failures": manifest.failures,
            "run": str(destination.resolve()),
            "runtime": dict(manifest.runtime),
            "output_bytes": sum(f.size_bytes for f in manifest.outputs),
        }
        (report_root / (job_name(job) + ".json")).write_bytes(canonical_json_bytes(summary))
        output.append(summary)
    return output
