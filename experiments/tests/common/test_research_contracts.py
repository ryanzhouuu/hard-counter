import pytest
from experiments.common.contracts import StudyConfig, Variant, fingerprint
from experiments.player_adjustment.support_contracts import SupportBins
from experiments.tests.common.test_research_matrix import frozen_config
from pydantic import ValidationError


def config() -> StudyConfig:
    return StudyConfig(study_id="synthetic", variants=(Variant(variant_id="A0"),))


def test_contract_hash_canonical_round_trip() -> None:
    value = config()
    restored = StudyConfig.model_validate_json(value.model_dump_json())
    assert fingerprint(value) == fingerprint(restored)
    assert restored == value
    with pytest.raises(ValidationError):
        StudyConfig.model_validate({**value.model_dump(), "unregistered": True})


@pytest.mark.parametrize(
    "stage", ["development-frozen", "candidate-frozen", "prospective-reporting"]
)
def test_stages_cannot_advance_without_explicit_decision(stage: str) -> None:
    with pytest.raises(ValidationError, match="advancement"):
        StudyConfig.model_validate({**config().model_dump(), "stage": stage})


def test_variants_reject_combined_studies_and_unexplained_gates() -> None:
    with pytest.raises(ValidationError, match="cannot combine"):
        Variant(variant_id="bad", feature_groups=("cycle",), nuisance="joint")
    with pytest.raises(ValidationError, match="reason"):
        Variant(variant_id="bad", enabled=False)
    with pytest.raises(ValidationError, match="unique IDs"):
        StudyConfig(study_id="bad", variants=(Variant(variant_id="a"), Variant(variant_id="a")))


def test_controlled_player_support_bins_are_required_and_hashed() -> None:
    payload = {**frozen_config().model_dump(), "study_id": "player-adjustment"}
    with pytest.raises(ValueError, match="preregistered support bins"):
        StudyConfig.model_validate(payload)
    payload["player_support_bins"] = SupportBins(
        prior_history_edges=(1, 10), deck_switching_edges=(1, 2)
    )
    declared = StudyConfig.model_validate(payload)
    changed = declared.model_copy(
        update={
            "player_support_bins": SupportBins(
                prior_history_edges=(1, 20), deck_switching_edges=(1, 2)
            )
        }
    )
    assert fingerprint(changed) != fingerprint(declared)
