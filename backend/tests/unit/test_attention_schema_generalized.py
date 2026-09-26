"""Expanded v1 snapshots encode additional identities without changing June defaults."""

from json import dumps

import pytest
from catalog_fixture import expanded_catalog
from pydantic import ValidationError

from clash_sos.domain import attention_schema
from clash_sos.domain.attention_schema import (
    AttentionCardSchema,
    AttentionModelConfig,
    build_attention_schema,
)


def test_expanded_schema_freezes_catalog_attributes_and_era() -> None:
    catalog, attributes = expanded_catalog()
    schema = build_attention_schema(
        catalog.serialize(),
        attributes=attributes,
        network=AttentionModelConfig(),
        balance_era_id="next-era",
    )
    assert schema.schema_version == "attention-card-schema:v1"
    assert schema.catalog_version == catalog.version
    assert schema.attribute_version == attributes.version
    assert schema.balance_era_id == "next-era"
    assert len(schema.identity_vocab) == 266
    assert len(schema.base_vocab) == 211
    tokens = schema.encode_deck(tuple(f"z-future-{index:03}:base" for index in range(82, 90)))
    assert max(tokens) > 255
    assert AttentionCardSchema.model_validate_json(schema.model_dump_json()) == schema
    pretty = build_attention_schema(
        dumps(catalog.to_payload(), indent=2).encode(),
        attributes=attributes,
        network=schema.network,
        balance_era_id=schema.balance_era_id,
    )
    assert pretty.fingerprint() == schema.fingerprint()


def test_schema_rejects_missing_attributes_and_extra_base_keys() -> None:
    catalog, attributes = expanded_catalog(1)
    attributes.cards.pop("z-future-000")
    with pytest.raises(ValueError, match="missing card attributes"):
        build_attention_schema(
            catalog.serialize(), attributes=attributes, network=AttentionModelConfig()
        )
    catalog, attributes = expanded_catalog(1)
    schema = build_attention_schema(
        catalog.serialize(), attributes=attributes, network=AttentionModelConfig()
    )
    payload = schema.model_dump(mode="python")
    payload["base_vocab"] = (*schema.base_vocab, "zz-unmapped")
    with pytest.raises(ValidationError, match="base vocabulary"):
        AttentionCardSchema.model_validate(payload)


def test_schema_checks_uint16_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog, attributes = expanded_catalog(1)
    monkeypatch.setattr(attention_schema, "MAX_CARD_IDENTITIES", 176)
    with pytest.raises(ValueError, match="uint16 capacity"):
        build_attention_schema(
            catalog.serialize(), attributes=attributes, network=AttentionModelConfig()
        )
