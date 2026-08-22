from datetime import UTC, datetime

from clash_sos.application.dataset_audit import BattleIdentityIndex
from clash_sos.domain.canonical import (
    Battle,
    BattleOutcome,
    BattleSide,
    CardForm,
    CardId,
    CardRef,
    Deck,
    PlayerId,
    RecordIssue,
    RecordState,
)


def battle() -> Battle:
    cards = tuple(
        CardRef(card_id=CardId(f"card-{index}"), form=CardForm.BASE) for index in range(16)
    )
    return Battle(
        side_a=BattleSide(
            player_id=PlayerId("#AAA"),
            deck=Deck(cards=cards[:8]),
            card_levels=(16,) * 8,
        ),
        side_b=BattleSide(
            player_id=PlayerId("#BBB"),
            deck=Deck(cards=cards[8:]),
            card_levels=(16,) * 8,
        ),
        outcome=BattleOutcome.SIDE_A_WIN,
        timestamp=datetime(2026, 6, 15, tzinfo=UTC),
        mode="Ranked1v1_NewArena",
        source_id="source:v1",
    )


def test_exact_duplicate_is_quarantined() -> None:
    index = BattleIdentityIndex()
    original = battle()

    assert index.observe(original, location="first.parquet:1") is None
    duplicate = index.observe(original, location="second.parquet:2")

    assert duplicate is not None
    assert duplicate.state is RecordState.QUARANTINED
    assert duplicate.issues == (RecordIssue.DUPLICATE_BATTLE,)
    assert duplicate.detail == "first observed at first.parquet:1"


def test_conflicting_outcome_is_quarantined_separately() -> None:
    index = BattleIdentityIndex()
    original = battle()
    conflict = original.model_copy(update={"outcome": BattleOutcome.SIDE_B_WIN})

    assert index.observe(original, location="first.parquet:1") is None
    disposition = index.observe(conflict, location="second.parquet:2")

    assert disposition is not None
    assert disposition.state is RecordState.QUARANTINED
    assert disposition.issues == (RecordIssue.CONFLICTING_BATTLE,)
