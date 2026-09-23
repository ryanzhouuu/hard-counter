"""Exercise score invariants across full and ablated attention networks."""

import pytest
import torch
from pydantic import ValidationError

from clash_sos.domain.attention_model import AttentionMatchupModel
from clash_sos.domain.attention_schema import AttentionModelConfig, build_attention_schema
from clash_sos.domain.card_attributes import CARD_ATTRIBUTES
from clash_sos.infrastructure.kaggle_v6.catalog import KAGGLE_V6_CARDS


def _model(**settings: object) -> AttentionMatchupModel:
    schema = build_attention_schema(
        KAGGLE_V6_CARDS.serialize(),
        attributes=CARD_ATTRIBUTES,
        network=AttentionModelConfig.model_validate(
            {"embedding_width": 16, "feed_forward_width": 32, **settings}
        ),
    )
    return AttentionMatchupModel(schema)


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"neural_component": False},
        {"explicit_interactions": False},
        {"within_deck_attention": False},
        {"cross_deck_attention": False},
    ],
)
def test_model_invariants_hold_before_and_after_training_step(settings: dict[str, bool]) -> None:
    torch.manual_seed(17)  # type: ignore[reportUnknownMemberType]
    model = _model(**settings)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    first = torch.arange(8)
    second = torch.arange(4, 12)
    rows = torch.stack((torch.stack((first, second)), torch.stack((second, first))))

    for _ in range(2):
        model.eval()
        with torch.inference_mode():
            logits = model(rows)
            swapped = model(rows.flip(1))
            same = model(torch.stack((first, first)).unsqueeze(0))
            permuted = model(rows[:, :, torch.tensor([7, 0, 6, 1, 5, 2, 4, 3])])
            single = torch.cat([model(row.unsqueeze(0)) for row in rows])
        torch.testing.assert_close(logits, -swapped, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            torch.sigmoid(logits) + torch.sigmoid(swapped),
            torch.ones_like(logits),
            atol=1e-5,
            rtol=1e-5,
        )
        assert torch.sigmoid(same).item() == pytest.approx(0.5, abs=1e-6)
        torch.testing.assert_close(logits, permuted, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(logits, single, atol=1e-5, rtol=1e-5)

        model.train()
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(rows), torch.tensor([1.0, 0.0])
        )
        loss.backward()  # type: ignore[reportUnknownMemberType]
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        optimizer.step()  # type: ignore[reportUnknownMemberType]


def test_disabled_paths_have_no_trainable_parameters() -> None:
    explicit = _model(neural_component=False)
    assert explicit.deck_encoder is None
    assert explicit.ordered_head is None
    assert not list(explicit.cross_blocks.parameters())
    assert set(dict(explicit.named_parameters())) == {
        "explicit.card_strength",
        "explicit.own_pair",
        "explicit.opposing_pair",
    }
    neural = _model(
        explicit_interactions=False, within_deck_attention=False, cross_deck_attention=False
    )
    assert neural.explicit is None
    assert neural.deck_encoder is not None
    assert not list(neural.deck_encoder.blocks.parameters())
    assert not list(neural.cross_blocks.parameters())
    with pytest.raises(ValidationError, match="at least one"):
        AttentionModelConfig(neural_component=False, explicit_interactions=False)


def test_model_rejects_wrong_input_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        _model()(torch.zeros(2, 8, dtype=torch.long))


def test_neural_model_overfits_a_support_dependent_counter_fixture() -> None:
    torch.manual_seed(23)  # type: ignore[reportUnknownMemberType]
    model = _model(explicit_interactions=False)
    rows: list[list[list[int]]] = []
    labels: list[float] = []
    for choice in (0, 1):
        for support in (2, 3):
            for opponent in (8, 9):
                rows.append([[choice, support, *range(20, 26)], [opponent, *range(40, 47)]])
                labels.append(float(((choice == 0) and (support == 2)) == (opponent == 8)))
    tokens = torch.tensor(rows)
    targets = torch.tensor(labels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    for _ in range(160):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(tokens), targets)
        loss.backward()  # type: ignore[reportUnknownMemberType]
        optimizer.step()  # type: ignore[reportUnknownMemberType]
    with torch.inference_mode():
        final_loss = torch.nn.functional.binary_cross_entropy_with_logits(model(tokens), targets)
    assert final_loss.item() < 0.1
