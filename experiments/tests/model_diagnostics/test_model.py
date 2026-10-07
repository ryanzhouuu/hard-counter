import pytest
import torch
from experiments.model_diagnostics.model import DiagnosticModel, ModelSpec, recipe
from experiments.player_adjustment.model import PlayerModel
from experiments.tests.player_adjustment.test_player_history import DECKS, row


def test_default_reproduces_existing_model_and_main_freezes_pairs() -> None:
    rows = (row(0), row(1))
    tokens = torch.tensor([DECKS])
    current = PlayerModel(11)
    equivalent = DiagnosticModel(11, ModelSpec(), rows)
    equivalent.load_state_dict(current.state_dict())
    torch.testing.assert_close(
        current(tokens, torch.empty(1, 0)), equivalent.matchup_logits(tokens)
    )
    main = DiagnosticModel(11, ModelSpec(capacity="main"), rows)
    with torch.no_grad():
        main.explicit.card_strength[0] = 0.7
    assert main.matchup_logits(tokens).item() == pytest.approx(0.7)
    assert not main.explicit.own_pair.requires_grad
    assert not main.explicit.opposing_pair.requires_grad
    assert sum(p.numel() for p in main.parameters() if p.requires_grad) == 11


@pytest.mark.parametrize("branch", ["history", "joint"])
def test_penalties_are_independent_and_skill_removal_is_exact(branch: str) -> None:
    rows = (row(0), row(1))
    spec = ModelSpec.model_validate(
        {"nuisance": branch, "card_l2": 2, "pair_l2": 3, "player_l2": 4}
    )
    model = DiagnosticModel(11, spec, rows)
    with torch.no_grad():
        model.explicit.card_strength[0] = 2
        model.explicit.own_pair[0] = 3
        model.explicit.opposing_pair[0] = 4
        model.beta.fill_(2)
        if model.player_effects is not None:
            model.player_effects.effects.copy_(torch.tensor([-2.0, 2.0]))
    expected_player = 16 if branch == "history" else 32
    assert model.penalty(0).item() == pytest.approx((8 + 75 + expected_player) / 2)
    tokens = torch.tensor([DECKS])
    features = torch.tensor([[1.0]]) if branch == "history" else torch.tensor([[0, 1]])
    neutral = torch.zeros_like(features) if branch == "history" else torch.full_like(features, -1)
    torch.testing.assert_close(model(tokens, neutral), model.matchup_logits(tokens))
    reverse = -features if branch == "history" else features.flip(1)
    torch.testing.assert_close(model(tokens.flip(1), reverse), -model(tokens, features))
    model.penalty(0).backward()  # type: ignore[reportUnknownMemberType]
    assert model.explicit.card_strength.grad is not None
    assert model.explicit.card_strength.grad[0].item() == 4
    with pytest.raises(ValueError, match="ModelSpec"):
        model.penalty(1)


@pytest.mark.parametrize(
    "payload",
    [
        {"card_l2": float("nan")},
        {"pair_l2": -1},
        {"capacity": "main", "pair_l2": 1},
        {"player_l2": 1},
    ],
)
def test_invalid_penalties_are_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ModelSpec.model_validate(payload)


def test_recipe_uses_training_only_for_history_and_unknown_players() -> None:
    rows = (row(0), row(1))
    later = (row(2, a="new", b="b"),)
    history, _ = recipe(ModelSpec(nuisance="history"), 11)
    assert history(rows, rows)[0, 0] == 0
    assert history(rows, later)[0, 0] > 0
    joint, factory = recipe(ModelSpec(nuisance="joint"), 11)
    assert joint(rows, later).tolist() == [[-1, 1]]
    assert isinstance(factory(rows, 2), DiagnosticModel)
