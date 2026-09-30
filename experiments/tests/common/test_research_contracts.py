import pytest
from experiments.common.contracts import StudyConfig, Variant, fingerprint
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
