from collections.abc import Callable, Sequence
from dataclasses import replace

import pytest
from experiments.form_mechanics.features import extract as form_extract
from experiments.higher_order.features import eligibility
from experiments.higher_order.features import extract as pattern_extract
from experiments.matchup_features.cycle import return_cost
from experiments.matchup_features.features import extract as matchup_extract
from experiments.matchup_features.responses import directed, summarize
from experiments.mechanics.contracts import (
    FeatureResult,
    MechanicsCatalog,
    MechanicsUnavailable,
    unknown,
)
from experiments.mechanics.load import synthetic_catalog
from experiments.tower_mechanics.features import extract as tower_extract

PAIR = [[0, 1, 2, 3, 4, 5, 6, 7, 13], [1, 2, 3, 4, 5, 6, 7, 10, 14]]
Extract = Callable[[Sequence[Sequence[int]], MechanicsCatalog], FeatureResult]


@pytest.mark.parametrize("extract", [matchup_extract, form_extract, tower_extract, pattern_extract])
def test_feature_swap_card_permutation_and_identical_lineups(extract: Extract) -> None:
    catalog = synthetic_catalog()
    original = extract(PAIR, catalog)
    swapped = extract(PAIR[::-1], catalog)
    permuted = [side[:8][::-1] + side[8:] for side in PAIR]
    assert original.values == tuple(-v for v in swapped.values)
    assert extract(permuted, catalog) == original
    assert all(value == 0 for value in extract([PAIR[0], PAIR[0]], catalog).values)
    assert len(original.names) == len(original.formulas)


def test_response_channels_pressure_and_cost_hand_calculation() -> None:
    catalog = synthetic_catalog()
    attackers = [catalog.for_token(i) for i in [1, 1, 6, 6, 6, 6, 6, 6, 13]]
    defenders = [catalog.for_token(i) for i in [2, 4, 6, 6, 6, 6, 6, 6, 13]]
    result = summarize(attackers, defenders, "multi_unit")
    assert result.threats == 2
    assert len(result.deployable) == len(result.spells) == 1
    assert len(result.towers) == 0
    assert result.minimum_cost == 2
    names, values, _ = directed(attackers, defenders)
    features = dict(zip(names, values, strict=True))
    assert features["response.multi_unit.sole_answer_pressure"] == 1
    assert features["response.multi_unit.response_context"] == 2
    defenders[0] = catalog.for_token(6)
    defenders[1] = catalog.for_token(6)
    result = summarize(attackers, defenders, "multi_unit")
    assert result.minimum_cost is None
    names, values, _ = directed(attackers, defenders)
    assert dict(zip(names, values, strict=True))["response.multi_unit.unanswered_exposure"] == 2


def test_cycle_excludes_threat_conditional_cards_and_tower() -> None:
    catalog = synthetic_catalog()
    cards = [catalog.for_token(i) for i in [6, 1, 4, 3, 2, 7, 8, 9, 13]]
    assert return_cost(cards, 0) == 2 + 2 + 3 + 4
    assert return_cost(cards, 1) == 1 + 2 + 3 + 4
    result = matchup_extract(PAIR, catalog, response=False, cost=False)
    assert result.names == result.values == ()


def test_form_inherited_vs_specific_same_registry_and_identical_descriptors() -> None:
    catalog = synthetic_catalog()
    specific = form_extract(PAIR, catalog)
    inherited = form_extract(PAIR, catalog, inherited=True)
    assert specific.names == inherited.names
    assert specific.values != inherited.values
    entries = dict(catalog.entries)
    entries[10] = replace(entries[10], fields=entries[6].fields)
    catalog = replace(catalog, entries=entries)
    assert form_extract(PAIR, catalog) == form_extract(PAIR, catalog, inherited=True)


def test_towers_four_identities_and_quantitative_missing_gate() -> None:
    catalog = synthetic_catalog()
    for tower in range(13, 17):
        pair = [[*PAIR[0][:-1], tower], PAIR[1]]
        assert len(tower_extract(pair, catalog, quantitative=True).names) > 28
    entries = dict(catalog.entries)
    fields = dict(entries[13].fields)
    fields["damage_per_hit"] = unknown("damage_per_hit")
    entries[13] = replace(entries[13], fields=fields)
    with pytest.raises(MechanicsUnavailable, match="damage_per_hit"):
        tower_extract(PAIR, replace(catalog, entries=entries), quantitative=True)


def test_pattern_positive_negative_cap_and_independent_answers() -> None:
    catalog = synthetic_catalog()
    attacking = [1, 1, 1, 4, 5, 6, 7, 0, 13]
    defending = [2, 6, 7, 8, 9, 0, 3, 5, 14]
    positive = pattern_extract([attacking, defending], catalog)
    assert positive.values[0] == 2
    assert positive.values[2] == 2
    defending[1] = 2
    negative = pattern_extract([attacking, defending], catalog)
    assert negative.values[0] == 0
    assert negative.values[2] == 0
    attacking[3] = 6
    no_spell = pattern_extract([attacking, defending], catalog)
    assert no_spell.values[0] == 0


def test_pattern_train_only_support_masks_keep_inactive_registry() -> None:
    catalog = synthetic_catalog()
    rows = [pattern_extract(PAIR, catalog), pattern_extract(PAIR[::-1], catalog)]
    mask = eligibility(rows, role="selection_fit", minimum_support=1)
    assert mask.names == rows[0].names
    assert len(mask.reasons) == 3
    with pytest.raises(ValueError, match="fit-role"):
        eligibility(rows, role="development", minimum_support=1)  # type: ignore[arg-type]
    empty = eligibility([rows[0], rows[0]], role="refit", minimum_support=1)
    assert not any(empty.active)
    assert set(empty.reasons) == {"constant_column"}


def test_higher_order_mixed_difference_cannot_be_a_sum_of_identity_and_pair_terms() -> None:
    from itertools import product

    catalog = synthetic_catalog()
    third_difference = 0.0
    for second_threat, spell_support, defensive_answer in product((0, 1), repeat=3):
        attacking = [1, 1 if second_threat else 6, 5 if spell_support else 6, 6, 6, 6, 6, 6, 13]
        defending = [2 if defensive_answer else 6, 6, 6, 6, 6, 6, 6, 6, 14]
        value = pattern_extract([attacking, defending], catalog).values[2]
        sign = (-1) ** (3 - second_threat - spell_support - defensive_answer)
        third_difference += sign * value
    assert third_difference == 2


def test_unknown_categorical_representation_is_finite_and_pattern_branch_is_gated() -> None:
    catalog = synthetic_catalog()
    entries = {}
    for token, entry in catalog.entries.items():
        fields = {
            name: field if field.status == "not_applicable" else unknown(name)
            for name, field in entry.fields.items()
        }
        entries[token] = replace(entry, fields=fields)
    unknown_catalog = replace(catalog, entries=entries)
    for extractor in (matchup_extract, form_extract, tower_extract):
        result = extractor(PAIR, unknown_catalog)
        assert result.names
        assert all(abs(v) < 1000 for v in result.values)
    with pytest.raises(MechanicsUnavailable, match="pattern field"):
        pattern_extract(PAIR, unknown_catalog)


def test_numeric_inapplicability_has_an_indicator_distinct_from_valid_zero() -> None:
    from experiments.mechanics.contracts import MechanicField

    catalog = synthetic_catalog()
    entries = dict(catalog.entries)
    fields = dict(entries[13].fields)
    fields["recharge"] = MechanicField(None, "seconds", "not_applicable")
    entries[13] = replace(entries[13], fields=fields)
    result = tower_extract(PAIR, replace(catalog, entries=entries), quantitative=True)
    lookup = dict(zip(result.names, result.values, strict=True))
    assert lookup["tower.recharge.not_applicable"] == 1
    assert lookup["tower.recharge.airborne"] == 0
