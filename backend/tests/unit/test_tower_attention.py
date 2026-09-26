"""Tower inputs affect matchup scores without breaking legacy score invariants."""

import pytest
import torch

from clash_sos.domain.attention_model import AttentionMatchupModel
from clash_sos.domain.attention_schema import (
    AttentionCardSchema,
    AttentionModelConfig,
    build_attention_schema,
)
from clash_sos.infrastructure.clash_royale.catalog import (
    CURRENT_CARD_ATTRIBUTES,
    CURRENT_CARD_CATALOG,
    CURRENT_TOWER_CATALOG,
)


def tower_schema(*, neural: bool = True) -> AttentionCardSchema:
    return build_attention_schema(
        CURRENT_CARD_CATALOG.serialize(),
        attributes=CURRENT_CARD_ATTRIBUTES,
        tower_catalog=CURRENT_TOWER_CATALOG,
        balance_era_id="2026-09",
        network=AttentionModelConfig(
            neural_component=neural,
            embedding_width=16,
            feed_forward_width=32,
        ),
    )


def test_v2_keeps_towers_outside_deck_and_freezes_their_lookup() -> None:
    schema = tower_schema()
    cards = tuple(entry.card.identity_key for entry in CURRENT_CARD_CATALOG.entries[:8])
    assert schema.input_size == 9 and len(schema.identity_vocab) == 186
    assert schema.schema_version == "attention-card-schema:v2"
    assert AttentionCardSchema.model_validate_json(schema.model_dump_json()) == schema
    encoded = schema.encode_side(cards, tower="cannoneer:tower", tower_level=16)
    assert encoded[:8] == schema.encode_deck(cards)
    assert encoded[8] == schema.identity_index["cannoneer:tower"]
    assert schema.token_attributes[encoded[8]] == (0,) * 8
    assert schema.encode_side(cards[::-1], tower="cannoneer:tower") == encoded
    for tower in (None, "future:tower", cards[0]):
        with pytest.raises(ValueError, match="known tower"):
            schema.encode_side(cards, tower=tower)
    with pytest.raises(ValueError, match="tower level 16"):
        schema.encode_side(cards, tower="cannoneer:tower", tower_level=15)
    with pytest.raises(ValueError, match="deck slots"):
        schema.encode_deck((*cards[:7], "cannoneer:tower"))


@pytest.mark.parametrize("neural", [False, True])
def test_tower_matchups_preserve_swapping_and_deck_permutation(neural: bool) -> None:
    schema = tower_schema(neural=neural)
    cards = tuple(entry.card.identity_key for entry in CURRENT_CARD_CATALOG.entries[:8])
    a = schema.encode_side(cards, tower="cannoneer:tower")
    b = schema.encode_side(cards, tower="dagger-duchess:tower")
    model = AttentionMatchupModel(schema)
    rows = torch.tensor([[a, b]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(2):
        model.eval()
        logits = model(rows)
        torch.testing.assert_close(logits, -model(rows.flip(1)), atol=1e-5, rtol=1e-5)
        permutation = torch.tensor([7, 0, 6, 1, 5, 2, 4, 3, 8])
        torch.testing.assert_close(logits, model(rows[:, :, permutation]), atol=1e-5, rtol=1e-5)
        assert model(torch.tensor([[a, a]])).item() == pytest.approx(0, abs=1e-6)
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones(1))
        loss.backward()  # type: ignore[reportUnknownMemberType]
        optimizer.step()  # type: ignore[reportUnknownMemberType]
    assert model(rows).item() != pytest.approx(0, abs=1e-4)
    with pytest.raises(ValueError, match="shape"):
        model(rows[:, :, :8])
