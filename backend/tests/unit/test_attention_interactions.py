"""Check exact explicit effects before they are mixed with neural scores."""

import pytest
import torch

from clash_sos.domain.attention_interactions import ExplicitInteractions


def test_explicit_score_has_checkable_card_pair_and_counter_terms() -> None:
    model = ExplicitInteractions(12)
    a = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
    b = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 8]])
    with torch.no_grad():
        model.card_strength[7] = 1.0
        model.own_pair[6] = 2.0  # (0, 7) in the 12-card upper triangle
        model.opposing_pair[56] = 3.0  # (7, 8)
    assert model(a, b).item() == pytest.approx(6.0)
    assert model(b, a).item() == pytest.approx(-6.0)
    assert model(a, a).item() == pytest.approx(0.0)
    assert model(a.flip(1), b.flip(1)).item() == pytest.approx(6.0)


def test_explicit_active_parameters_receive_gradients() -> None:
    model = ExplicitInteractions(12)
    a = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
    b = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 8]])
    model(a, b).sum().backward()
    assert model.card_strength.grad is not None
    assert model.own_pair.grad is not None
    assert model.opposing_pair.grad is not None
    assert model.card_strength.grad[7] != 0
    assert model.own_pair.grad.abs().sum() > 0
    assert model.opposing_pair.grad.abs().sum() > 0


def test_explicit_rejects_bad_shapes() -> None:
    model = ExplicitInteractions(12)
    with pytest.raises(ValueError, match="matching"):
        model(torch.zeros(1, 8, dtype=torch.long), torch.zeros(2, 8, dtype=torch.long))
