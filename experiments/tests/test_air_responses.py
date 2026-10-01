"""Air answers require a verified response channel, not an unknown spell type."""

from dataclasses import replace

import pytest
from experiments.matchup_features.responses import answers, directed, summarize
from experiments.mechanics.contracts import unknown
from experiments.mechanics.load import synthetic_catalog


def test_unknown_spell_classification_is_not_a_deployable_air_answer() -> None:
    entry = synthetic_catalog().for_token(2)
    candidate = replace(entry, fields={**entry.fields, "spell": unknown("spell")})
    assert candidate.flag("targets_air")
    assert not answers(candidate, "airborne")
    summary = summarize([entry] * 8, [candidate] * 8, "airborne")
    assert summary.deployable == summary.spells == ()
    assert summary.unknown == 8


def test_unknown_spell_classification_is_not_a_deployable_area_answer() -> None:
    entry = synthetic_catalog().for_token(2)
    candidate = replace(entry, fields={**entry.fields, "spell": unknown("spell")})
    summary = summarize([entry] * 8, [candidate] * 8, "multi_unit")
    assert summary.deployable == summary.spells == ()
    assert summary.unknown == 8


@pytest.mark.parametrize("answer_count", [0, 1, 2])
def test_air_scarcity_and_sole_answer_pressure_by_hand(answer_count: int) -> None:
    catalog = synthetic_catalog()
    attacker = catalog.for_token(0)
    attacker = replace(
        attacker,
        fields={**attacker.fields, "airborne": replace(attacker.field("airborne"), value=True)},
    )
    empty = catalog.for_token(6)
    ground_tower = replace(
        catalog.for_token(13),
        fields={
            **catalog.for_token(13).fields,
            "targets_air": replace(catalog.for_token(13).field("targets_air"), value=False),
        },
    )
    attackers = [attacker, attacker, *([empty] * 6), ground_tower]
    defenders = [
        *([catalog.for_token(2)] * answer_count),
        *([empty] * (8 - answer_count)),
        ground_tower,
    ]
    names, values, _ = directed(attackers, defenders)
    result = dict(zip(names, values, strict=True))
    assert result["response.airborne.threat_count"] == 2
    assert result["response.airborne.deployable_answers"] == answer_count
    assert result["response.airborne.unanswered_exposure"] == (2 if answer_count == 0 else 0)
    assert result["response.airborne.sole_answer_pressure"] == (1 if answer_count == 1 else 0)


def test_tower_answer_does_not_need_a_spell_or_deploy_cost() -> None:
    tower = synthetic_catalog().for_token(13)
    tower = replace(tower, fields={**tower.fields, "spell": unknown("spell")})
    assert answers(tower, "airborne")
    summary = summarize([], [tower], "airborne")
    assert summary.towers == (tower,)
    assert summary.deployable == summary.spells == ()
    assert summary.minimum_cost is None
