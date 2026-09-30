from dataclasses import replace

import pytest
import torch
from experiments.common.attention_reference import (
    AttentionReference,
    input_encoding_digest,
    require_identical_reference_population,
)
from experiments.common.data_access import ResearchRow
from experiments.matchup_features.model import ResearchModel
from test_tower_attention import tower_schema
from tower_dataset_fixture import tower_rows


def test_attention_reference_preserves_native_score_and_checkpoint() -> None:
    schema = tower_schema()
    model = AttentionReference(schema, expected_schema_sha256=schema.fingerprint())
    cards = tuple(
        schema.identity_index[identity]
        for identity in schema.identity_vocab
        if not identity.endswith(":tower")
    )[:8]
    towers = sorted(schema.tower_indices)
    tokens = torch.tensor([[(*cards, towers[0]), (*cards, towers[1])]])
    features = torch.empty((1, 0))
    torch.testing.assert_close(model(tokens, features), model.reference(tokens))
    torch.testing.assert_close(model(tokens.flip(1), features), -model(tokens, features))
    assert model.parameter_groups(0.02)[0]["weight_decay"] == 0.02
    assert len(list(model.parameters())) == len(list(model.reference.parameters()))
    restored = AttentionReference(schema, expected_schema_sha256=schema.fingerprint())
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(tokens, features), model(tokens, features))
    with pytest.raises(ValueError, match="network-specific"):
        AttentionReference(schema, expected_schema_sha256="0" * 64)
    with pytest.raises(ValueError, match="features"):
        model(tokens, torch.zeros((1, 1)))


def test_disabled_neural_reference_matches_research_explicit_baseline() -> None:
    schema = tower_schema(neural=False)
    model = AttentionReference(schema, expected_schema_sha256=schema.fingerprint())
    baseline = ResearchModel(len(schema.identity_vocab), 0)
    assert model.reference.explicit is not None
    with torch.no_grad():
        baseline.explicit.card_strength.copy_(torch.arange(len(schema.identity_vocab)) / 20)
    model.reference.explicit.load_state_dict(baseline.explicit.state_dict())
    indices = [
        index for index in range(len(schema.identity_vocab)) if index not in schema.tower_indices
    ]
    towers = sorted(schema.tower_indices)
    tokens = torch.tensor([[(*indices[:8], towers[0]), (*indices[4:12], towers[1])]])
    features = torch.empty((1, 0))
    torch.testing.assert_close(model(tokens, features), baseline(tokens, features))
    assert model.penalty(0.01).item() == 0


def test_reference_cache_comparison_checks_population_without_erasing_architecture() -> None:
    explicit, attention = tower_schema(neural=False), tower_schema()
    assert explicit.fingerprint() != attention.fingerprint()
    assert input_encoding_digest(explicit) == input_encoding_digest(attention)
    source = tower_rows()[0]
    tokens = (
        attention.encode_side(
            tuple(
                f"{card}:{form}"
                for card, form in zip(source.side_a_card_ids, source.side_a_card_forms, strict=True)
            ),
            tower=source.side_a_tower,
        ),
        attention.encode_side(
            tuple(
                f"{card}:{form}"
                for card, form in zip(source.side_b_card_ids, source.side_b_card_forms, strict=True)
            ),
            tower=source.side_b_tower,
        ),
    )
    row = ResearchRow((source.timestamp, "a", "b", 0), source.event_key, "a", "b", 1, tokens)
    require_identical_reference_population(explicit, (row,), attention, (row,))
    for changed in (row.swapped(), replace(row, label=0), replace(row, event_key="changed")):
        with pytest.raises(ValueError, match="oriented"):
            require_identical_reference_population(explicit, (row,), attention, (changed,))
