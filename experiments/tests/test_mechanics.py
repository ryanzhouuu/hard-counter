from dataclasses import replace
from json import dumps
from pathlib import Path
from urllib.parse import urlparse

import pytest
from experiments.mechanics.contracts import (
    MechanicField,
    MechanicsCatalog,
    MechanicsEntry,
    decode_pair,
    unknown,
)
from experiments.mechanics.load import from_payload, load, synthetic_catalog, to_payload


def test_catalog_round_trip_digest_and_read_only_fields(tmp_path: Path) -> None:
    catalog = synthetic_catalog()
    path = tmp_path / "mechanics.json"
    path.write_text(dumps(to_payload(catalog)))
    assert load(path, expected_digest=catalog.digest).digest == catalog.digest
    with pytest.raises(ValueError, match="digest"):
        load(path, expected_digest="wrong")
    with pytest.raises(TypeError):
        catalog.entries[0] = catalog.entries[1]  # type: ignore[index]
    changed = dict(catalog.entries)
    changed[0] = replace(changed[0], fields={"airborne": MechanicField(False, "flag", "synthetic")})
    assert replace(catalog, entries=changed).digest != catalog.digest


def test_false_unknown_inapplicable_and_measured_zero_are_distinct() -> None:
    entry = synthetic_catalog().for_token(6)
    assert entry.field("airborne").value is False
    assert unknown("recharge").status == "unknown"
    assert synthetic_catalog().for_token(13).field("deploy_cost").status == "not_applicable"
    assert entry.number("ability_cost") == 0


@pytest.mark.parametrize(
    "field",
    [
        lambda: MechanicField(None, "flag", "verified"),
        lambda: MechanicField(True, "flag", "verified", "http://example.test", "2026-09-01"),
        lambda: MechanicField(0, "damage", "synthetic", level=14),
        lambda: MechanicField(0, "flag", "unknown"),
    ],
)
def test_invalid_evidence_rejected(field: object) -> None:
    with pytest.raises(ValueError):
        field()  # type: ignore[operator]


def test_unit_level_era_and_tower_contracts() -> None:
    with pytest.raises(ValueError, match="unit"):
        MechanicsEntry("x:base", "card", {"deploy_cost": MechanicField(2, "seconds", "synthetic")})
    with pytest.raises(ValueError, match="level"):
        MechanicsEntry(
            "x:tower", "card", {"damage_per_hit": MechanicField(2, "damage", "synthetic")}
        )
    with pytest.raises(ValueError, match="inapplicable"):
        MechanicsEntry("x:tower", "tower", {})
    with pytest.raises(ValueError, match="era"):
        replace(synthetic_catalog(), era_start="2026-10-01")
    with pytest.raises(ValueError, match="synthetic"):
        replace(synthetic_catalog(), synthetic=False)


def test_inheritance_is_explicit_and_override_preserved() -> None:
    catalog = synthetic_catalog()
    form = catalog.entries[10]
    fields = dict(form.fields)
    fields["deploy_cost"] = replace(fields["deploy_cost"], inherited_from=form.base_identity)
    entries = dict(catalog.entries)
    entries[10] = replace(form, fields=fields)
    catalog = replace(catalog, entries=entries)
    assert catalog.for_token(10).number("deploy_cost") == 1
    assert catalog.for_token(10).flag("area_damage")
    assert not catalog.for_token(10, inherited=True).flag("area_damage")
    fields["deploy_cost"] = replace(fields["deploy_cost"], inherited_from="fixture-0:base")
    entries[10] = replace(form, fields=fields)
    with pytest.raises(ValueError, match="inheritance"):
        replace(catalog, entries=entries)


def test_duplicate_tokens_identities_and_unknown_schema_rejected() -> None:
    payload = to_payload(synthetic_catalog())
    payload["surprise"] = 1
    with pytest.raises(ValueError, match="fields"):
        from_payload(payload)
    catalog = synthetic_catalog()
    with pytest.raises(ValueError, match="duplicate"):
        MechanicsCatalog(
            "v", "2026-09-01", "2026-09-30", {0: catalog.entries[0], 1: catalog.entries[0]}, True
        )


def test_slots_and_coverage() -> None:
    catalog = synthetic_catalog()
    pair = [[0, 1, 2, 3, 4, 5, 6, 7, 13], [0, 1, 2, 3, 4, 5, 6, 7, 16]]
    left, right = decode_pair(pair, catalog)
    assert left[-1].identity == "tower-princess:tower"
    assert right[-1].identity == "royal-chef:tower"
    pair[0][0], pair[0][8] = pair[0][8], pair[0][0]
    with pytest.raises(ValueError, match="ninth"):
        decode_pair(pair, catalog)
    assert catalog.coverage([0], ["airborne"])["fixture-0:base"]["airborne"] == "synthetic"


def test_sourced_partial_inventory_covers_official_forms_and_gates_quantitative() -> None:
    from experiments.mechanics.load import bind_tokens, load_partial_catalog
    from experiments.tower_mechanics.features import require_quantitative

    catalog = load_partial_catalog()
    assert not catalog.synthetic
    assert any(e.identity.endswith(":evolution") for e in catalog.entries.values())
    assert any(e.identity.endswith(":hero") for e in catalog.entries.values())
    towers = [token for token, e in catalog.entries.items() if e.kind == "tower"]
    assert len(towers) == 4
    for entry in catalog.entries.values():
        for field in entry.fields.values():
            if field.known:
                if field.status == "user_reported":
                    assert field.source_url is None and field.evidence_sha256
                    assert field.source_effective_date
                    continue
                assert field.status == "verified"
                assert field.source_url
                if urlparse(field.source_url).hostname == "royaleapi.com":
                    assert urlparse(field.source_url).path.startswith(("/blog/", "/card/"))
                elif urlparse(field.source_url).hostname == "clashroyale.fandom.com":
                    assert urlparse(field.source_url).path.startswith("/wiki/")
                    assert field.source_effective_date == "2026-10-01"
                else:
                    assert urlparse(field.source_url).hostname in {
                        "supercell.com",
                        "support.clashroyale.com",
                    }
                assert field.source_effective_date
    with pytest.raises(ValueError, match="unavailable"):
        require_quantitative(catalog, towers)
    rebound = bind_tokens(catalog, ("royal-chef:tower", "spirit-empress:base"))
    assert rebound.entries[0].identity == "royal-chef:tower"
    assert rebound.entries[1].field("cost_kind").value == "conditional"
    assert rebound.entries[1].number("deploy_cost") is None
    assert rebound.entries[1].field("airborne").status == "unknown"
