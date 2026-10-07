from dataclasses import replace

import numpy as np
import pytest
import torch
from experiments.common.calibration import fit_temperature, sigmoid
from experiments.common.data_access import ResearchRow
from experiments.model_diagnostics.model import DiagnosticModel, ModelSpec
from experiments.tests.player_adjustment.test_player_history import DECKS, row


def sampled_population(seed: int, count: int, offset: int = 0) -> tuple[ResearchRow, ...]:
    rng = np.random.default_rng(seed)
    skill = np.linspace(-1.2, 1.2, 8)
    rows: list[ResearchRow] = []
    for i in range(count):
        a = int(rng.integers(8))
        b = (a + int(rng.integers(1, 8))) % 8
        sparse = rng.random() < 0.9
        skill_b = 0 if sparse else skill[b]
        deck_a = int(rng.random() < sigmoid(float(1.5 * skill[a])))
        deck_b = int(rng.random() < sigmoid(float(1.5 * skill_b)))
        logit = 0.8 * (deck_b - deck_a) + skill[a] - skill_b
        rows.append(
            replace(
                row(
                    i + offset,
                    a=f"core{a}",
                    b=f"leaf{i + offset}" if sparse else f"core{b}",
                    label=int(rng.random() < sigmoid(float(logit))),
                ),
                tokens=(DECKS[deck_a], DECKS[deck_b]),
            )
        )
    return tuple(rows)


def test_binary_sparse_correlated_data_recovers_matchup_after_skill_removal() -> None:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        training = sampled_population(17, 6000)
        model = DiagnosticModel(
            11, ModelSpec(capacity="main", nuisance="joint", player_l2=0.001), training
        )
        assert model.player_effects is not None
        tokens = torch.tensor([r.tokens for r in training])
        players = torch.tensor(model.player_effects.vocabulary.encode(training))
        labels = torch.tensor([r.label for r in training], dtype=torch.float32)
        optimizer = torch.optim.Adam(model.parameter_groups(0), lr=0.025)
        for _ in range(180):
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(tokens, players), labels
            )
            (loss + model.penalty(0)).backward()  # type: ignore[reportUnknownMemberType]
            optimizer.step()  # type: ignore[reportUnknownMemberType]
            model.center_nuisance()
        calibration = sampled_population(19, 2500, offset=6000)
        with torch.inference_mode():
            logits = model(
                torch.tensor([r.tokens for r in calibration]),
                torch.tensor(model.player_effects.vocabulary.encode(calibration)),
            ).numpy()
            matchup = float(model.matchup_logits(torch.tensor([DECKS])).item())
        temperature = fit_temperature(
            tuple(map(float, logits)), tuple(r.label for r in calibration)
        )
        assert matchup == pytest.approx(0.8, abs=0.2)
        assert matchup / temperature.temperature == pytest.approx(0.8, abs=0.25)
        assert len(model.player_effects.vocabulary.players) > 5000
    finally:
        torch.set_num_threads(previous)


def test_fixed_player_decks_admit_identical_outcomes_with_different_matchups() -> None:
    rows = tuple(row(i) for i in range(12))
    first = DiagnosticModel(11, ModelSpec(nuisance="joint"), rows)
    second = DiagnosticModel(11, ModelSpec(nuisance="joint"), rows)
    assert first.player_effects is not None and second.player_effects is not None
    with torch.no_grad():
        first.explicit.card_strength[0] = 0.8
        second.player_effects.effects.copy_(torch.tensor([0.4, -0.4]))
    tokens = torch.tensor([r.tokens for r in rows])
    players = torch.tensor(first.player_effects.vocabulary.encode(rows))
    torch.testing.assert_close(first(tokens, players), second(tokens, players))
    assert not torch.allclose(first.matchup_logits(tokens), second.matchup_logits(tokens))
