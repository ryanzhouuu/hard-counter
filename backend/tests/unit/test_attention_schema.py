"""Lock the June 2026 token vocabulary and reject incompatible deck inputs."""

import os
import subprocess
import sys
from copy import deepcopy
from json import dumps, loads
from pathlib import Path

import pytest
from pydantic import ValidationError

from clash_sos.domain.attention_schema import (
    ATTRIBUTE_COLUMNS,
    ELIXIR_DIVISOR,
    AttentionCardSchema,
    AttentionModelConfig,
    build_attention_schema,
)
from clash_sos.domain.card_attributes import CARD_ATTRIBUTES
from clash_sos.infrastructure.kaggle_v6.catalog import KAGGLE_V6_CARDS


@pytest.fixture
def schema() -> AttentionCardSchema:
    """Build against the same versioned inputs used by the processed dataset."""
    return build_attention_schema(
        KAGGLE_V6_CARDS.serialize(),
        attributes=CARD_ATTRIBUTES,
        network=AttentionModelConfig(),
    )


def test_schema_snapshots_every_identity_and_round_trips(schema: AttentionCardSchema) -> None:
    catalog_identities = {entry.card.identity_key for entry in KAGGLE_V6_CARDS.entries}
    assert len(schema.identity_vocab) == 176
    assert len(schema.base_vocab) == 121
    assert schema.form_vocab == ("base", "champion", "evolution", "hero")
    assert set(schema.identity_vocab) == catalog_identities
    assert schema.identity_vocab == tuple(sorted(catalog_identities))
    assert schema.attribute_columns == ATTRIBUTE_COLUMNS
    assert schema.elixir_divisor == ELIXIR_DIVISOR == 9
    assert schema.network.embedding_width == 64
    restored = AttentionCardSchema.model_validate(schema.model_dump(mode="python"))
    assert restored.fingerprint() == schema.fingerprint()
    assert restored.encode_deck(schema.identity_vocab[:8]) == tuple(range(8))


def test_base_and_form_lookup_keeps_mirror_missingness(schema: AttentionCardSchema) -> None:
    knight = schema.identity_index["knight:base"]
    evolved = schema.identity_index["knight:evolution"]
    mirror = schema.identity_index["mirror:base"]
    xbow = schema.identity_index["x-bow:base"]
    assert schema.token_base_indices[knight] == schema.token_base_indices[evolved]
    assert schema.token_form_indices[knight] != schema.token_form_indices[evolved]
    assert schema.token_attributes[knight] == schema.token_attributes[evolved]
    assert schema.token_attributes[mirror][:2] == (0.0, 1.0)
    assert schema.token_attributes[xbow] == pytest.approx((6 / 9, 0, 1, 1, 0, 0, 0, 0))


def test_deck_encoding_ignores_card_order_and_validates_level(schema: AttentionCardSchema) -> None:
    cards = schema.identity_vocab[:8]
    expected = schema.encode_deck(cards, levels=(16,) * 8)
    assert schema.encode_deck(tuple(reversed(cards))) == expected
    assert schema.encode_deck(cards) == expected
    with pytest.raises(ValueError, match="level 16"):
        schema.encode_deck(cards, levels=(16,) * 7 + (15,))
    with pytest.raises(ValueError, match="eight aligned"):
        schema.encode_deck(cards[:7])
    with pytest.raises(ValueError, match="eight aligned"):
        schema.encode_deck(cards, levels=(16,) * 7)
    with pytest.raises(ValueError, match="distinct"):
        schema.encode_deck(cards[:7] + cards[:1])
    with pytest.raises(ValueError, match="unsupported card identity"):
        schema.encode_deck((*cards[:7], "not-a-card:base"))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "attention-card-schema:v3", "Input should be"),
        ("attribute_version", "card-attributes:2026-07", "attribute snapshot"),
        ("catalog_version", "catalog:next", "catalog version"),
        ("catalog_sha256", "0" * 64, "catalog hash"),
        ("token_base_indices", (-1,), "lookup arrays"),
    ],
)
def test_schema_rejects_incompatible_snapshots(
    schema: AttentionCardSchema, field: str, value: object, message: str
) -> None:
    payload = deepcopy(schema.model_dump(mode="python"))
    payload[field] = value
    with pytest.raises(ValidationError, match=message):
        AttentionCardSchema.model_validate(payload)


def test_schema_rejects_corrupted_lookup_and_attributes(schema: AttentionCardSchema) -> None:
    payload = deepcopy(schema.model_dump(mode="python"))
    indices = list(payload["token_base_indices"])
    indices[0] = len(schema.base_vocab)
    payload["token_base_indices"] = indices
    with pytest.raises(ValidationError, match="out of range"):
        AttentionCardSchema.model_validate(payload)

    payload = deepcopy(schema.model_dump(mode="python"))
    vectors = list(payload["token_attributes"])
    vectors[0] = (0.0,) * 8
    payload["token_attributes"] = vectors
    with pytest.raises(ValidationError, match="token lookup"):
        AttentionCardSchema.model_validate(payload)

    payload = deepcopy(schema.model_dump(mode="python"))
    attributes = payload["attribute_snapshot"]
    attributes["cards"].pop("knight")
    with pytest.raises(ValidationError, match="base vocabulary"):
        AttentionCardSchema.model_validate(payload)


def test_model_config_rejects_nondivisible_attention_width() -> None:
    with pytest.raises(ValidationError, match="divisible"):
        AttentionModelConfig(embedding_width=65, attention_heads=4)
    with pytest.raises(ValidationError, match="within-deck attention requires"):
        AttentionModelConfig(within_deck_blocks=0)
    assert AttentionModelConfig(within_deck_blocks=0, within_deck_attention=False)


def test_model_configuration_changes_schema_identity(schema: AttentionCardSchema) -> None:
    wider = build_attention_schema(
        KAGGLE_V6_CARDS.serialize(),
        attributes=CARD_ATTRIBUTES,
        network=AttentionModelConfig(embedding_width=128),
    )
    assert wider.fingerprint() != schema.fingerprint()


def test_default_switch_preserves_cache_identity(schema: AttentionCardSchema) -> None:
    assert schema.fingerprint() == (
        "1127cc7ccda8447fc8c597d9a44edf1dc63be16423ccc87c4e92246d43082b6b"
    )
    explicit = build_attention_schema(
        KAGGLE_V6_CARDS.serialize(),
        attributes=CARD_ATTRIBUTES,
        network=AttentionModelConfig(neural_component=False),
    )
    assert explicit.fingerprint() != schema.fingerprint()


def test_builder_rejects_corrupted_catalog_snapshots() -> None:
    catalog = loads(KAGGLE_V6_CARDS.serialize())
    catalog["entries"][1][1] = catalog["entries"][0][1]
    with pytest.raises(ValueError, match="unique"):
        build_attention_schema(
            dumps(catalog).encode(),
            attributes=CARD_ATTRIBUTES,
            network=AttentionModelConfig(),
        )


def test_schema_and_protocol_import_without_torch() -> None:
    backend_src = Path(__file__).parents[2] / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(backend_src), environment.get("PYTHONPATH")) if value
    )
    script = """
import builtins
real_import = builtins.__import__
def reject_torch(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch" or name.startswith("torch."):
        raise ModuleNotFoundError("torch blocked", name="torch")
    return real_import(name, globals, locals, fromlist, level)
builtins.__import__ = reject_torch
import clash_sos.domain.attention_schema
import clash_sos.domain.attention_protocol
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    assert result.returncode == 0, result.stderr
