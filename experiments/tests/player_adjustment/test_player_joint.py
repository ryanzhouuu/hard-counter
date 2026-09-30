from dataclasses import replace

import numpy as np
import pytest
import torch
from experiments.common.data_access import ResearchRow
from experiments.player_adjustment.diagnostics import training_diagnostics
from experiments.player_adjustment.joint import FrozenPlayerEffects, JointPlayerEffects
from experiments.player_adjustment.model import PlayerModel
from experiments.tests.player_adjustment.test_player_history import DECKS, row
from torch.nn import functional as F


def test_joint_unknown_centering_penalty_and_serialization() -> None:
    rows = [row(0), row(1, a="c", b="d")]
    joint = JointPlayerEffects.from_training(rows)
    with torch.no_grad():
        joint.effects.copy_(torch.tensor([1.0, 3.0, 8.0, 4.0]))
    encoded = torch.tensor(joint.vocabulary.encode(rows))
    before = joint(encoded)
    joint.center_()
    torch.testing.assert_close(joint(encoded), before)
    torch.testing.assert_close(joint.effects, torch.tensor([-1.0, 1.0, 2.0, -2.0]))
    assert joint.penalty(2).item() == pytest.approx(2.5)
    unseen = joint.vocabulary.encode([row(2, a="new", b="a")])
    assert unseen.tolist() == [[-1, 0]]
    assert joint(torch.tensor(unseen)).item() == 1
    frozen = FrozenPlayerEffects.model_validate_json(joint.freeze().model_dump_json())
    restored = JointPlayerEffects.from_frozen(frozen)
    torch.testing.assert_close(restored(encoded), before)
    for invalid in (torch.tensor([[0.2, 1.0]]), torch.tensor([[-2, 0]])):
        with pytest.raises(ValueError, match=r"indices|index"):
            joint(invalid)


@pytest.mark.parametrize("branch", ["history", "joint"])
def test_player_model_swap_and_matchup_interpretations(branch: str) -> None:
    rows = [row(0), row(1)]
    joint = JointPlayerEffects.from_training(rows)
    model = PlayerModel(
        11,
        branch="history" if branch == "history" else "joint",
        player_effects=None if branch == "history" else joint,
    )
    tokens = torch.tensor([DECKS])
    features = torch.tensor([[1.2]]) if branch == "history" else torch.tensor([[0, 1]])
    reverse = -features if branch == "history" else features.flip(1)
    with torch.no_grad():
        model.explicit.card_strength[0] = 0.2
        model.beta.fill_(2)
        joint.effects.copy_(torch.tensor([-0.4, 0.4]))
    torch.testing.assert_close(model(tokens.flip(1), reverse), -model(tokens, features))
    assert model(tokens, features).item() != model.matchup_logits(tokens).item()
    assert model.parameter_groups(0.01)[1]["weight_decay"] == 0
    restored = PlayerModel(
        11,
        branch="history" if branch == "history" else "joint",
        player_effects=None if branch == "history" else JointPlayerEffects.from_training(rows),
    )
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(tokens, features), model(tokens, features))
    with torch.no_grad():
        model.beta.zero_()
        joint.effects.zero_()
    torch.testing.assert_close(model(tokens, features), model.matchup_logits(tokens))


def test_crossing_deck_assignments_recover_simulated_player_effects() -> None:
    rows: list[ResearchRow] = []
    true_effects = torch.tensor([-0.8, -0.3, 0.3, 0.8])
    for left in range(4):
        for right in range(left + 1, 4):
            for repetition in range(12):
                base = row(len(rows), a=f"p{left}", b=f"p{right}")
                rows.append(replace(base, tokens=DECKS if repetition % 2 else DECKS[::-1]))
    joint = JointPlayerEffects.from_training(rows)
    model = PlayerModel(11, branch="joint", player_effects=joint)
    tokens = torch.tensor([item.tokens for item in rows])
    indices = torch.tensor(joint.vocabulary.encode(rows))
    deck_gap = torch.tensor([0.4 if item.tokens == DECKS else -0.4 for item in rows])
    target = torch.sigmoid(deck_gap + true_effects[indices[:, 0]] - true_effects[indices[:, 1]])
    optimizer = torch.optim.Adam(model.parameter_groups(0), lr=0.04)
    for _ in range(180):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(tokens, indices), target)
        loss.backward()  # type: ignore[reportUnknownMemberType]
        optimizer.step()  # type: ignore[reportUnknownMemberType]
        model.center_nuisance()
    np.testing.assert_allclose(joint.freeze().effects, true_effects.numpy(), atol=0.015)
    assert training_diagnostics(rows).single_deck_players == 0
    confounded = [
        replace(
            item, tokens=(DECKS[int(item.player_a[-1]) // 2], DECKS[int(item.player_b[-1]) // 2])
        )
        for item in rows
    ]
    diagnostics = training_diagnostics(confounded)
    assert diagnostics.single_deck_players == 4
    assert any("confounding" in warning for warning in diagnostics.warnings)
