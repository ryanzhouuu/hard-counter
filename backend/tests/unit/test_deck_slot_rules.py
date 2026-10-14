"""Matchup decks follow the Evolution, Hero, and Wild slot limits."""

import pytest
from pydantic import ValidationError

from clash_sos.application.deck_matchup import MatchupDeck

ROOTS = ["a", "b", "c", "d", "e", "f", "g", "h"]


def deck(*forms: str) -> tuple[str, ...]:
    padded = [*forms, *["base"] * (8 - len(forms))]
    return tuple(f"{root}:{form}" for root, form in zip(ROOTS, padded, strict=True))


@pytest.mark.parametrize(
    "forms",
    [
        (),
        ("evolution", "evolution", "hero"),
        ("champion", "champion", "evolution"),
        ("hero", "champion"),
    ],
)
def test_legal_slot_use_is_accepted(forms: tuple[str, ...]) -> None:
    assert len(MatchupDeck(cards=deck(*forms)).cards) == 8


@pytest.mark.parametrize(
    "forms",
    [
        ("evolution", "evolution", "evolution"),
        ("hero", "hero", "champion"),
        ("evolution", "evolution", "hero", "champion"),
    ],
)
def test_slot_overflow_is_rejected(forms: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError, match="at most 2 Evolutions"):
        MatchupDeck(cards=deck(*forms))
