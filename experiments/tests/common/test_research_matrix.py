from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
from experiments.common.artifacts import file_record, finish_run
from experiments.common.contracts import (
    ComparisonRules,
    FileRecord,
    PopulationIdentity,
    Role,
    RunManifest,
    StudyConfig,
    Variant,
    fingerprint,
)
from experiments.common.data_access import Operation, ResearchRow, RoleAccess
from experiments.common.matrix import (
    DevelopmentResult,
    MatrixJob,
    confirmation_jobs,
    development_loss,
    screen_jobs,
    select_penalties,
    smoke_jobs,
)
from experiments.common.predictions import Prediction, write_predictions
from experiments.tests.common.research_fixture import HASH, START, protocol, rows

from clash_sos.domain.attention_protocol import digest_row_keys


def matrix_config(study: str, count: int, *, attention: bool = False) -> StudyConfig:
    variants = [Variant(variant_id="baseline")]
    variants.extend(
        Variant(variant_id=f"candidate-{index}", feature_groups=("features",))
        for index in range(count)
    )
    if attention:
        variants.append(Variant(variant_id="full_attention", architecture="attention"))
    return StudyConfig(study_id=study, variants=tuple(variants))


def frozen_config() -> StudyConfig:
    population_rows = rows()
    population = PopulationIdentity(
        snapshot_files=(FileRecord(path="source.jsonl", sha256=HASH, size_bytes=1),),
        row_keys_sha256=digest_row_keys(item.key for item in population_rows),
        event_mapping_sha256=HASH,
        oriented_sha256=HASH,
        mode="synthetic",
        era="synthetic",
        mirror_seed=0,
        start=START,
        end=population_rows[-1].key[0] + timedelta(hours=1),
        catalog_sha256=HASH,
    )
    return StudyConfig(
        study_id="higher-order",
        stage="development-frozen",
        variants=matrix_config("higher-order", 1).variants,
        population=population,
        decision_sha256=HASH,
        schema_sha256=HASH,
        feature_definitions_sha256=HASH,
        rules=ComparisonRules(practical_margin=0.001, maximum_brier_regression=0.001),
    )


class DevelopmentAccessSpy(RoleAccess):
    def __init__(self) -> None:
        population = rows()
        super().__init__(protocol(population), population)
        self.reads: list[tuple[Role, Operation]] = []

    def read(self, role: Role, operation: Operation) -> tuple[ResearchRow, ...]:
        self.reads.append((role, operation))
        assert role == "development" and operation == "compare"
        return super().read(role, operation)


def result_run(
    root: Path,
    config: StudyConfig,
    job: MatrixJob,
    confidence: float,
    *,
    corrupt_job: bool = False,
    eligible: bool = True,
) -> DevelopmentResult:
    destination = root / f"{job.variant_id}-{job.penalty}"
    destination.mkdir()
    access = DevelopmentAccessSpy()
    predictions = tuple(
        Prediction(
            item.key,
            item.event_key,
            item.key[0],
            item.player_a,
            item.player_b,
            item.label,
            0.0,
            confidence if item.label else 1 - confidence,
        )
        for item in access.read("development", "compare")
    )
    write_predictions(destination / "development.json", predictions)
    population = config.population
    assert population is not None
    manifest = RunManifest(
        run_id="synthetic",
        config=config,
        config_sha256=fingerprint(config),
        git_sha="b" * 40,
        lock_sha256=HASH,
        stage=config.stage,
        status="complete",
        eligible_for_comparison=eligible,
        inputs=population.snapshot_files,
        outputs=(file_record(destination / "development.json", destination),),
        population=population,
        runtime=(
            ("code_sha256", HASH),
            ("variant_id", "wrong" if corrupt_job else job.variant_id),
            ("seed", str(job.seed)),
            ("penalty", str(job.penalty)),
            ("phase", job.phase),
        ),
    )
    finish_run(destination, manifest)
    return DevelopmentResult(job, destination, "development.json")


@pytest.mark.parametrize(
    "study,count,attention,expected",
    [
        ("response-cycle", 3, True, 21),
        ("player-adjustment", 2, False, 10),
        ("form-mechanics", 2, False, 10),
        ("tower-mechanics", 2, False, 10),
        ("tower-mechanics", 1, False, 5),
        ("higher-order", 1, False, 5),
    ],
)
def test_registered_new_fit_totals(study: str, count: int, attention: bool, expected: int) -> None:
    config = matrix_config(study, count, attention=attention)
    selected = {f"candidate-{index}": config.penalties[0] for index in range(count)}
    jobs = (*screen_jobs(config), *confirmation_jobs(config, selected))
    assert sum(not job.shared for job in jobs) == expected
    assert len(smoke_jobs(config)) == len(config.variants)
    assert all(job.seed == 0 and job.phase == "smoke" for job in smoke_jobs(config))
    assert all(job.phase == "reference" for job in jobs if job.variant_id == "full_attention")


def test_disabled_branches_and_unregistered_confirmation_settings() -> None:
    config = matrix_config("tower-mechanics", 2)
    payload = config.model_dump(mode="json")
    payload["variants"][-1] = Variant(
        variant_id="candidate-1", enabled=False, disabled_reason="unsourced"
    ).model_dump()
    gated = StudyConfig.model_validate(payload)
    assert all(job.variant_id != "candidate-1" for job in smoke_jobs(gated))
    with pytest.raises(ValueError, match="registered penalty"):
        confirmation_jobs(gated, {"candidate-0": 42})


def test_selection_reads_verified_development_grid_only(tmp_path: Path) -> None:
    config = frozen_config()
    access = DevelopmentAccessSpy()
    jobs = [job for job in screen_jobs(config) if job.penalty]
    results = [
        result_run(tmp_path, config, job, confidence)
        for job, confidence in zip(jobs, (0.8, 0.95, 0.85), strict=True)
    ]
    assert select_penalties(config, results, access) == {"candidate-0": config.penalties[1]}
    assert set(access.reads) == {("development", "compare")}
    assert config.stage == "development-frozen"
    with pytest.raises(ValueError, match="exactly once"):
        select_penalties(config, [results[0]], access)
    with pytest.raises(ValueError, match="registered"):
        development_loss(config, replace(results[0], job=replace(jobs[0], penalty=42)), access)
    smoke = matrix_config("higher-order", 1)
    untouched = DevelopmentAccessSpy()
    with pytest.raises(ValueError, match="smoke"):
        select_penalties(smoke, results, untouched)
    assert not untouched.reads
    (results[0].directory / "development.json").write_text("corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        development_loss(config, results[0], access)


@pytest.mark.parametrize("corrupt_job,eligible", [(True, True), (False, False)])
def test_selection_rejects_wrong_job_or_ineligible_run(
    tmp_path: Path, corrupt_job: bool, eligible: bool
) -> None:
    config = frozen_config()
    job = screen_jobs(config)[1]
    result = result_run(tmp_path, config, job, 0.8, corrupt_job=corrupt_job, eligible=eligible)
    with pytest.raises(ValueError, match=r"registered job|eligible"):
        development_loss(config, result, DevelopmentAccessSpy())
