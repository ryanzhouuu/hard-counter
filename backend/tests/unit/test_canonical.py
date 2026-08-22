from datetime import UTC, datetime, timedelta, timezone

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import AnyHttpUrl, ValidationError

from clash_sos.domain.canonical import (
    BalanceChange,
    BalanceEra,
    BalanceEraRegistry,
    Battle,
    BattleOutcome,
    BattleSide,
    CardForm,
    CardId,
    CardRef,
    Deck,
    EraBoundaryEvidence,
    EraBoundaryPolicy,
    PlayerId,
    RecordDisposition,
    RecordIssue,
    RecordState,
)


def card(card_id: str, form: CardForm = CardForm.BASE) -> CardRef:
    return CardRef(card_id=CardId(card_id), form=form)


def deck(offset: int = 0) -> Deck:
    return Deck(
        cards=tuple(card(f"card-{index + offset}") for index in range(8)),
    )


def evidence(
    stated_date: datetime, policy: EraBoundaryPolicy = EraBoundaryPolicy.AT_STATED_DATE
) -> EraBoundaryEvidence:
    return EraBoundaryEvidence(
        summary="A documented balance update.",
        reference=AnyHttpUrl("https://example.test/balance-notes/june"),
        stated_date=stated_date,
        precision="day",
        boundary_policy=policy,
    )


def era() -> BalanceEra:
    return BalanceEra(
        era_id="2026-06",
        valid_from=datetime(2026, 6, 1, tzinfo=UTC),
        valid_to=datetime(2026, 7, 1, tzinfo=UTC),
        card_catalog_version="2026-06-v1",
        changed_cards=(
            BalanceChange(card_id=CardId("knight"), change_type="buff", summary="More hitpoints"),
        ),
        start_evidence=evidence(
            datetime(2026, 5, 31, tzinfo=UTC), EraBoundaryPolicy.DAY_AFTER_STATED_DATE
        ),
        end_evidence=evidence(datetime(2026, 7, 1, tzinfo=UTC)),
    )


def july_era() -> BalanceEra:
    return BalanceEra(
        era_id="2026-07",
        valid_from=datetime(2026, 7, 1, tzinfo=UTC),
        valid_to=datetime(2026, 8, 1, tzinfo=UTC),
        card_catalog_version="2026-06-v1",
        start_evidence=evidence(datetime(2026, 7, 1, tzinfo=UTC)),
        end_evidence=evidence(datetime(2026, 8, 1, tzinfo=UTC)),
    )


def overlapping_era() -> BalanceEra:
    return BalanceEra(
        era_id="2026-06-overlap",
        valid_from=datetime(2026, 6, 15, tzinfo=UTC),
        valid_to=datetime(2026, 7, 1, tzinfo=UTC),
        card_catalog_version="2026-06-v1",
        start_evidence=evidence(datetime(2026, 6, 15, tzinfo=UTC)),
        end_evidence=evidence(datetime(2026, 7, 1, tzinfo=UTC)),
    )


def open_ended_era() -> BalanceEra:
    return BalanceEra(
        era_id="2026-06-open",
        valid_from=datetime(2026, 6, 1, tzinfo=UTC),
        card_catalog_version="2026-06-v1",
        start_evidence=evidence(datetime(2026, 6, 1, tzinfo=UTC)),
    )


def side(player: str, battle_deck: Deck | None = None) -> BattleSide:
    return BattleSide(
        player_id=PlayerId(player),
        deck=battle_deck or deck(),
        card_levels=(16,) * 8,
    )


def test_card_ids_are_stable_form_independent_catalog_keys() -> None:
    assert CardId("royal-giant").value == "royal-giant"
    assert card("knight", CardForm.EVOLUTION).identity_key == "knight:evolution"


@pytest.mark.parametrize("value", ["Royal-Giant", "royal giant", "royal_.giant", ""])
def test_card_id_rejects_display_names_and_non_keys(value: str) -> None:
    with pytest.raises(ValidationError):
        CardId(value)


def test_player_tag_is_trimmed_and_uppercased() -> None:
    assert PlayerId("  #ab12cd ").value == "#AB12CD"


def test_deck_preserves_order_but_hashes_order_independently() -> None:
    first = deck()
    second = Deck(cards=tuple(reversed(first.cards)))

    assert first.cards[0].card_id == CardId("card-0")
    assert first.canonical_hash == second.canonical_hash


def test_deck_allows_same_card_in_distinct_forms() -> None:
    cards = [card(f"card-{index}") for index in range(6)]
    cards.extend([card("knight", CardForm.BASE), card("knight", CardForm.EVOLUTION)])

    assert len(Deck(cards=tuple(cards)).cards) == 8


@pytest.mark.parametrize(
    "cards, message",
    [
        (tuple(card(f"card-{index}") for index in range(7)), "exactly eight"),
        (tuple(card("knight") for _ in range(8)), "unique"),
    ],
)
def test_deck_rejects_incomplete_or_repeated_cards(
    cards: tuple[CardRef, ...], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        Deck(cards=cards)


def test_outcome_swapping_inverts_sides() -> None:
    assert BattleOutcome.SIDE_A_WIN.swapped() is BattleOutcome.SIDE_B_WIN
    assert BattleOutcome.DRAW.swapped() is BattleOutcome.DRAW


def test_balance_era_uses_half_open_interval() -> None:
    current = era()

    assert current.contains(datetime(2026, 6, 30, 23, 59, tzinfo=UTC))
    assert not current.contains(datetime(2026, 7, 1, tzinfo=UTC))


def test_balance_era_requires_provenance_for_finite_boundaries() -> None:
    with pytest.raises(ValidationError, match="start_evidence"):
        BalanceEra.model_validate(
            {
                "era_id": "2026-06",
                "valid_from": datetime(2026, 6, 1, tzinfo=UTC),
                "valid_to": datetime(2026, 7, 1, tzinfo=UTC),
                "card_catalog_version": "2026-06-v1",
            }
        )

    with pytest.raises(ValidationError, match="end-evidence"):
        BalanceEra.model_validate(
            {
                "era_id": "2026-06",
                "valid_from": datetime(2026, 6, 1, tzinfo=UTC),
                "valid_to": datetime(2026, 7, 15, tzinfo=UTC),
                "card_catalog_version": "2026-06-v1",
                "start_evidence": evidence(datetime(2026, 6, 1, tzinfo=UTC)).model_dump(
                    mode="json"
                ),
            }
        )


def test_battle_requires_aligned_card_levels() -> None:
    battle = Battle(
        side_a=side("#AAA111"),
        side_b=side("#BBB222", deck(offset=8)),
        outcome=BattleOutcome.SIDE_A_WIN,
        timestamp=datetime(2026, 6, 15, tzinfo=UTC),
        mode="Ranked1v1_NewArena",
        source_id="kaggle:source:v6",
    )

    assert battle.side_a.card_levels == (16,) * 8


def test_battle_requires_distinct_players() -> None:
    with pytest.raises(ValidationError, match="distinct players"):
        Battle(
            side_a=side("#AAA111"),
            side_b=side("#AAA111", deck(offset=8)),
            outcome=BattleOutcome.SIDE_A_WIN,
            timestamp=datetime(2026, 6, 15, tzinfo=UTC),
            mode="Ranked1v1_NewArena",
            source_id="kaggle:source:v6",
        )


def test_balance_era_boundaries_must_align_with_evidence() -> None:
    base = era().model_dump(mode="json")

    with pytest.raises(ValidationError, match="valid_from must align"):
        BalanceEra.model_validate(
            {**base, "valid_from": datetime(2026, 6, 2, tzinfo=UTC)},
        )

    with pytest.raises(ValidationError, match="valid_to must align"):
        BalanceEra.model_validate({**base, "valid_to": datetime(2026, 7, 2, tzinfo=UTC)})


def test_registry_lookup_respects_half_open_edges() -> None:
    registry = BalanceEraRegistry(registry_version="test:v1", eras=(era(),))

    assert registry.lookup(datetime(2026, 6, 1, tzinfo=UTC)) == era()
    assert registry.lookup(datetime(2026, 6, 30, 23, 59, tzinfo=UTC)) == era()
    assert registry.lookup(datetime(2026, 5, 31, 23, 59, tzinfo=UTC)) is None
    assert registry.lookup(datetime(2026, 7, 1, tzinfo=UTC)) is None


def test_registry_rejects_naive_timestamps() -> None:
    registry = BalanceEraRegistry(registry_version="test:v1", eras=(era(),))

    with pytest.raises(ValueError, match="timezone-aware"):
        registry.lookup(datetime(2026, 6, 15))


def test_registry_allows_adjacent_half_open_intervals() -> None:
    registry = BalanceEraRegistry(registry_version="test:v1", eras=(era(), july_era()))
    boundary = datetime(2026, 7, 1, tzinfo=UTC)

    assert registry.lookup(boundary - timedelta(microseconds=1)) == era()
    assert registry.lookup(boundary) == july_era()


def test_registry_rejects_overlapping_intervals() -> None:
    with pytest.raises(ValidationError, match="overlap"):
        BalanceEraRegistry(registry_version="test:v1", eras=(era(), overlapping_era()))


def test_registry_rejects_an_open_ended_interval_followed_by_another() -> None:
    with pytest.raises(ValidationError, match="overlap"):
        BalanceEraRegistry(registry_version="test:v1", eras=(open_ended_era(), era()))


def test_registry_rejects_duplicate_era_ids() -> None:
    with pytest.raises(ValidationError, match="unique"):
        BalanceEraRegistry(registry_version="test:v1", eras=(era(), era()))


def test_registry_supports_open_ended_final_intervals() -> None:
    open_ended = open_ended_era()
    registry = BalanceEraRegistry(registry_version="test:v1", eras=(open_ended,))

    assert registry.lookup(datetime(2030, 1, 1, tzinfo=UTC)) is open_ended


def test_empty_registry_assigns_no_era() -> None:
    registry = BalanceEraRegistry(registry_version="empty:v1")

    assert registry.eras == ()
    assert registry.lookup(datetime(2026, 6, 15, tzinfo=UTC)) is None


def test_registry_normalizes_era_order_for_deterministic_serialization() -> None:
    forward = BalanceEraRegistry(registry_version="test:v1", eras=(era(), july_era()))
    reversed_input = BalanceEraRegistry(registry_version="test:v1", eras=(july_era(), era()))

    assert forward.eras == reversed_input.eras
    assert forward.model_dump_json() == reversed_input.model_dump_json()


@given(st.sampled_from(tuple(BattleOutcome)))
def test_battle_fingerprint_is_independent_of_side_orientation(
    outcome: BattleOutcome,
) -> None:
    battle = Battle(
        side_a=side("#AAA111"),
        side_b=side("#BBB222", deck(offset=8)),
        outcome=outcome,
        timestamp=datetime(2026, 6, 15, tzinfo=UTC),
        mode="Ranked1v1_NewArena",
        source_id="kaggle:source:v6:row-1",
    )
    swapped = battle.model_copy(
        update={
            "side_a": battle.side_b,
            "side_b": battle.side_a,
            "outcome": battle.outcome.swapped(),
        }
    )

    assert battle.fingerprint == swapped.fingerprint
    assert battle.event_key == swapped.event_key


@given(st.permutations(tuple(range(8))))
def test_battle_identity_preserves_card_level_alignment_across_deck_order(
    permutation: tuple[int, ...],
) -> None:
    first_side = BattleSide(
        player_id=PlayerId("#AAA111"),
        deck=deck(),
        card_levels=tuple(range(9, 17)),
    )
    reordered_side = BattleSide(
        player_id=first_side.player_id,
        deck=Deck(cards=tuple(first_side.deck.cards[index] for index in permutation)),
        card_levels=tuple(first_side.card_levels[index] for index in permutation),
    )
    battle = Battle(
        side_a=first_side,
        side_b=side("#BBB222", deck(offset=8)),
        outcome=BattleOutcome.SIDE_A_WIN,
        timestamp=datetime(2026, 6, 15, tzinfo=UTC),
        mode="Ranked1v1_NewArena",
        source_id="kaggle:source:v6",
    )

    assert battle.fingerprint == battle.model_copy(update={"side_a": reordered_side}).fingerprint


@given(st.integers(min_value=-12, max_value=14))
def test_battle_identity_normalizes_equivalent_timestamp_offsets(offset: int) -> None:
    battle = Battle(
        side_a=side("#AAA111"),
        side_b=side("#BBB222", deck(offset=8)),
        outcome=BattleOutcome.SIDE_A_WIN,
        timestamp=datetime(2026, 6, 15, 12, tzinfo=UTC),
        mode="Ranked1v1_NewArena",
        source_id="kaggle:source:v6",
    )
    equivalent = battle.model_copy(
        update={"timestamp": battle.timestamp.astimezone(timezone(timedelta(hours=offset)))}
    )

    assert battle.fingerprint == equivalent.fingerprint
    assert battle.event_key == equivalent.event_key


@given(st.sampled_from((BattleOutcome.SIDE_A_WIN, BattleOutcome.SIDE_B_WIN)))
def test_battle_fingerprint_distinguishes_conflicting_outcomes(
    outcome: BattleOutcome,
) -> None:
    battle = Battle(
        side_a=side("#AAA111"),
        side_b=side("#BBB222", deck(offset=8)),
        outcome=outcome,
        timestamp=datetime(2026, 6, 15, tzinfo=UTC),
        mode="Ranked1v1_NewArena",
        source_id="kaggle:source:v6",
    )
    conflict = battle.model_copy(update={"outcome": outcome.swapped()})

    assert battle.event_key == conflict.event_key
    assert battle.fingerprint != conflict.fingerprint


def test_battle_rejects_naive_timestamps() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        Battle(
            side_a=side("#AAA111"),
            side_b=side("#BBB222", deck(offset=8)),
            outcome=BattleOutcome.SIDE_A_WIN,
            timestamp=datetime(2026, 6, 15),
            mode="Ranked1v1_NewArena",
            source_id="kaggle:source:v6",
        )


def test_record_disposition_makes_pipeline_states_explicit() -> None:
    for state, issue in (
        (RecordState.INVALID, RecordIssue.MALFORMED_TIMESTAMP),
        (RecordState.QUARANTINED, RecordIssue.UNKNOWN_CARD),
        (RecordState.UNSUPPORTED, RecordIssue.UNSUPPORTED_MODE),
        (RecordState.INSUFFICIENT_DATA, RecordIssue.INSUFFICIENT_HISTORY),
    ):
        disposition = RecordDisposition(state=state, issues=(issue,))
        assert disposition.state is state

    assert RecordDisposition(state=RecordState.VALID).issues == ()


def test_non_valid_disposition_requires_issue() -> None:
    with pytest.raises(ValidationError, match="at least one issue"):
        RecordDisposition(state=RecordState.QUARANTINED)


def test_side_rejects_misaligned_or_non_max_card_levels() -> None:
    with pytest.raises(ValidationError, match="align"):
        BattleSide(player_id=PlayerId("#AAA111"), deck=deck(), card_levels=(16,) * 7)

    with pytest.raises(ValidationError, match="between 1 and 16"):
        BattleSide(player_id=PlayerId("#AAA111"), deck=deck(), card_levels=(17,) * 8)
