"""Inventoried provenance for completed and failed prospective evaluations."""

from pathlib import Path

from experiments.common.artifacts import completed_manifest, file_record, finish_run
from experiments.common.candidate import CandidateFreeze
from experiments.common.contracts import FileRecord, RunManifest, StudyConfig, fingerprint
from experiments.common.data_access import ReportingContract
from experiments.common.provenance import code_digest, revision, versions
from experiments.common.reporting_consumption import registry_path


def finish_confirmation(
    directory: Path,
    run_id: str,
    config: StudyConfig,
    contract: ReportingContract,
    frozen: CandidateFreeze,
    freeze_path: Path,
    contract_path: Path,
    failure: Exception | None,
) -> None:
    git_sha, dirty, lock = revision(Path.cwd())
    outputs = tuple(
        file_record(path, directory) for path in sorted(directory.rglob("*")) if path.is_file()
    )
    manifest = completed_manifest(
        run_id,
        config,
        contract.population,
        git_sha,
        dirty,
        lock,
        "failed" if failure else "complete",
        outputs,
        (
            *versions(),
            ("code_sha256", code_digest(Path.cwd())),
            ("candidate_sha256", fingerprint(frozen)),
            ("candidate_source", str(freeze_path.resolve())),
            ("operation", "prospective-reporting-no-fit"),
            ("consumption_registry", str(registry_path(Path.cwd()).resolve())),
            ("consumption_identity", contract.population.row_keys_sha256),
        ),
        (),
        (),
        (),
        (str(failure),) if failure else (),
    )
    inputs = [*manifest.inputs]
    for path, name in ((freeze_path, "candidate-freeze.json"), (contract_path, "contract.json")):
        member = file_record(path, path.parent)
        inputs.append(
            FileRecord.model_validate({**member.model_dump(), "path": "candidate/" + name})
        )
    inputs.extend(
        FileRecord.model_validate(
            {**member.model_dump(), "path": "candidate/assets/" + member.path}
        )
        for member in frozen.assets
    )
    finish_run(
        directory,
        RunManifest.model_validate({**manifest.model_dump(), "inputs": tuple(inputs)}),
    )
