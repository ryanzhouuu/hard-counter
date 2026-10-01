"""Persist one development penalty decision and verify exact resumes."""

from pathlib import Path

from clash_sos.domain.canonical_dataset import canonical_json_bytes
from clash_sos.domain.manifests import ManifestModel, Sha256
from experiments.common.artifacts import file_record, load_run
from experiments.common.contracts import PopulationIdentity, StudyConfig, fingerprint
from experiments.common.data_access import RoleAccess
from experiments.common.matrix import DevelopmentResult, confirmation_jobs, select_penalties
from experiments.common.provenance import code_digest, revision


class SelectionDecision(ManifestModel):
    config_sha256: Sha256
    population: PopulationIdentity
    code_sha256: Sha256
    git_sha: str
    lock_sha256: Sha256
    screening_manifests: tuple[tuple[str, Sha256], ...]
    selected_penalties: tuple[tuple[str, float], ...]


def select_once(
    config: StudyConfig,
    results: list[DevelopmentResult],
    access: RoleAccess,
    destination: Path,
    root: Path,
) -> dict[str, float]:
    """Resume a recorded choice without reading development outcomes again."""
    if config.stage != "development-frozen" or config.population is None:
        raise PermissionError("selection requires frozen development inputs")
    git_sha, dirty, lock = revision(root)
    code = code_digest(root)
    if dirty:
        raise ValueError("controlled selection requires a clean tracked checkout")
    inventory: list[tuple[str, str]] = []
    for result in results:
        run = load_run(result.directory)
        if (run.git_sha, run.lock_sha256, dict(run.runtime).get("code_sha256")) != (
            git_sha,
            lock,
            code,
        ):
            raise ValueError("screening runs must share the current code and dependency lock")
        inventory.append(
            (
                str(result.directory.resolve()),
                file_record(result.directory / "manifest.json", result.directory).sha256,
            )
        )
    expected = {
        "config_sha256": fingerprint(config),
        "population": config.population,
        "code_sha256": code,
        "git_sha": git_sha,
        "lock_sha256": lock,
        "screening_manifests": tuple(sorted(inventory)),
    }
    if destination.exists():
        recorded = SelectionDecision.model_validate_json(destination.read_bytes())
        if any(getattr(recorded, key) != value for key, value in expected.items()):
            raise ValueError(
                "selection resume requires identical configuration and screening assets"
            )
        selected = dict(recorded.selected_penalties)
        confirmation_jobs(config, selected)
        return selected
    selected = select_penalties(config, results, access)
    recorded = SelectionDecision.model_validate(
        {**expected, "selected_penalties": tuple(sorted(selected.items()))}
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(canonical_json_bytes(recorded.model_dump()))
    return selected
