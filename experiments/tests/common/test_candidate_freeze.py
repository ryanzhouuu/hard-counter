from datetime import timedelta

import pytest
from experiments.common.candidate import (
    CandidateFreeze,
    freeze_candidate,
    validate_candidate_population,
)
from experiments.common.contracts import FileRecord, PopulationIdentity, StudyConfig, Variant
from experiments.tests.common.research_fixture import HASH
from experiments.tests.common.test_research_matrix import frozen_config


def candidate() -> CandidateFreeze:
    config = frozen_config()
    assert config.population is not None
    end = config.population.end
    return freeze_candidate(
        config,
        {"candidate-0": config.penalties[0]},
        code_sha256=HASH,
        assets=(FileRecord(path="model.pt", sha256=HASH, size_bytes=10),),
        inspected_through=end,
        frozen_at=end + timedelta(hours=1),
        reporting_start=end + timedelta(hours=2),
    )


def prospective(frozen: CandidateFreeze) -> PopulationIdentity:
    payload = frozen.fit_population.model_dump()
    payload.update(
        start=frozen.reporting_start,
        end=frozen.reporting_start + timedelta(days=1),
        snapshot_files=(FileRecord(path="future.jsonl", sha256="b" * 64, size_bytes=2),),
    )
    return PopulationIdentity.model_validate(payload)


def test_candidate_freeze_keeps_source_provenance_and_does_not_promote_stages() -> None:
    config = frozen_config()
    frozen = candidate()
    assert frozen.fit_population == config.population
    assert frozen.selections[0].seeds == config.seeds
    assert config.stage == "development-frozen"
    assert CandidateFreeze.model_validate_json(frozen.model_dump_json()) == frozen
    validate_candidate_population(frozen, prospective(frozen))
    payload = frozen.model_dump()
    payload["reporting_start"] = frozen.frozen_at
    with pytest.raises(ValueError, match="strictly after"):
        CandidateFreeze.model_validate(payload)


def test_candidate_rejects_backfills_era_changes_and_reused_sources() -> None:
    frozen = candidate()
    valid = prospective(frozen)
    for change, message in (
        ({"start": frozen.reporting_start - timedelta(minutes=1)}, "backfilled"),
        ({"era": "changed"}, "incompatible"),
        ({"snapshot_files": frozen.fit_population.snapshot_files}, "separate"),
    ):
        payload = valid.model_dump()
        payload.update(change)
        with pytest.raises(ValueError, match=message):
            validate_candidate_population(frozen, PopulationIdentity.model_validate(payload))


def test_final_freeze_can_select_one_challenger_from_a_larger_registered_study() -> None:
    config = frozen_config()
    payload = config.model_dump()
    payload["variants"] = (
        *config.variants,
        Variant(variant_id="candidate-1", feature_groups=("x",)),
    )
    expanded = StudyConfig.model_validate(payload)
    earlier = candidate()
    frozen = freeze_candidate(
        expanded,
        {"candidate-0": expanded.penalties[0]},
        code_sha256=HASH,
        assets=earlier.assets,
        frozen_at=earlier.frozen_at,
        inspected_through=earlier.inspected_through,
        reporting_start=earlier.reporting_start,
    )
    assert [item.variant_id for item in frozen.selections] == ["candidate-0"]
