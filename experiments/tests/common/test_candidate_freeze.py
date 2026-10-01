from datetime import timedelta

import pytest
from experiments.common.candidate import (
    CandidateFreeze,
    freeze_candidate,
    validate_candidate_population,
)
from experiments.common.contracts import (
    ComparisonRules,
    FileRecord,
    PopulationIdentity,
    StudyConfig,
    Variant,
)
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
        confirmation_rules=config.rules.model_copy(
            update={"confirmation": "single", "test_alternative": "two_sided"}
        ),
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
        confirmation_rules=earlier.confirmation_rules,
    )
    assert [item.variant_id for item in frozen.selections] == ["candidate-0"]


def test_candidate_freezes_the_complete_confirmation_rules() -> None:
    frozen = candidate()
    assert frozen.confirmation_rules.confirmation == "single"
    assert frozen.confirmation_rules.test_alternative == "two_sided"
    assert frozen.comparison_ids == ("candidate-0",)


def test_attention_comparison_cannot_hide_in_a_single_test_design() -> None:
    frozen = candidate()
    payload = frozen.model_dump()
    payload["comparison_ids"] = (*frozen.comparison_ids, "full_attention")
    with pytest.raises(ValueError, match=r"single.*one comparison"):
        CandidateFreeze.model_validate(payload)
    payload["confirmation_rules"] = ComparisonRules(
        confirmation="holm", test_alternative="improvement"
    )
    expanded = CandidateFreeze.model_validate(payload)
    assert len(expanded.comparison_ids) == 2
    payload["comparison_ids"] = (*expanded.comparison_ids, "unregistered")
    with pytest.raises(ValueError, match="comparison family"):
        CandidateFreeze.model_validate(payload)


def test_freeze_requires_explicit_rules_and_registered_attention() -> None:
    frozen = candidate()
    payload = frozen.model_dump()
    payload["confirmation_rules"] = ComparisonRules()
    with pytest.raises(ValueError, match="preregistered confirmation rules"):
        CandidateFreeze.model_validate(payload)
    del payload["confirmation_rules"]
    with pytest.raises(ValueError, match="confirmation_rules"):
        CandidateFreeze.model_validate(payload)
    config = frozen_config()

    def with_attention(source: StudyConfig) -> CandidateFreeze:
        return freeze_candidate(
            source,
            {"candidate-0": source.penalties[0]},
            code_sha256=HASH,
            assets=frozen.assets,
            frozen_at=frozen.frozen_at,
            inspected_through=frozen.inspected_through,
            reporting_start=frozen.reporting_start,
            confirmation_rules=frozen.confirmation_rules.model_copy(
                update={"confirmation": "holm"}
            ),
            include_attention=True,
        )

    with pytest.raises(ValueError, match="registered enabled reference"):
        with_attention(config)
    config = StudyConfig.model_validate(
        {
            **config.model_dump(),
            "variants": (
                *config.variants,
                Variant(variant_id="full_attention", architecture="attention"),
            ),
        }
    )
    declared = with_attention(config)
    assert declared.comparison_ids == ("candidate-0", "full_attention")
