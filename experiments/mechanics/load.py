"""Strict serialization and synthetic fixtures for the research mechanics catalog."""

from dataclasses import asdict
from json import loads
from pathlib import Path
from typing import Literal, cast

from experiments.mechanics.contracts import (
    FIELD_UNITS,
    MechanicField,
    MechanicsCatalog,
    MechanicsEntry,
    Status,
    Value,
)


def _field_payload(field: MechanicField) -> dict[str, object]:
    """Keep old catalog digests stable when a field has no manual evidence."""
    payload = asdict(field)
    if field.evidence_sha256 is None:
        payload.pop("evidence_sha256")
    return payload


def to_payload(catalog: MechanicsCatalog) -> dict[str, object]:
    return {
        "version": catalog.version,
        "era_start": catalog.era_start,
        "era_end": catalog.era_end,
        "synthetic": catalog.synthetic,
        "entries": [
            {
                "token": token,
                "identity": entry.identity,
                "kind": entry.kind,
                "base_identity": entry.base_identity,
                "fields": {
                    name: _field_payload(field) for name, field in sorted(entry.fields.items())
                },
            }
            for token, entry in sorted(catalog.entries.items())
        ],
    }


def from_payload(payload: object) -> MechanicsCatalog:
    if not isinstance(payload, dict):
        raise ValueError("mechanics catalog must be an object")
    body = cast(dict[str, object], payload)
    if set(body) != {"version", "era_start", "era_end", "synthetic", "entries"}:
        raise ValueError("invalid mechanics catalog fields")
    if any(not isinstance(body[name], str) for name in ("version", "era_start", "era_end")):
        raise ValueError("catalog version and era must be strings")
    if type(body["synthetic"]) is not bool or not isinstance(body["entries"], list):
        raise ValueError("invalid synthetic marker or entries")
    entries: dict[int, MechanicsEntry] = {}
    for raw in cast(list[object], body["entries"]):
        if not isinstance(raw, dict):
            raise ValueError("mechanics entry must be an object")
        entry = cast(dict[str, object], raw)
        if set(entry) != {"token", "identity", "kind", "base_identity", "fields"}:
            raise ValueError("invalid mechanics entry fields")
        token = entry["token"]
        if type(token) is not int or token in entries:
            raise ValueError("invalid or duplicate mechanics token")
        if not isinstance(entry["identity"], str) or entry["kind"] not in {"card", "tower"}:
            raise ValueError("invalid mechanics identity/kind")
        if entry["base_identity"] is not None and not isinstance(entry["base_identity"], str):
            raise ValueError("invalid form base")
        if not isinstance(entry["fields"], dict):
            raise ValueError("mechanics fields must be an object")
        fields: dict[str, MechanicField] = {}
        for name, raw_field in cast(dict[object, object], entry["fields"]).items():
            if not isinstance(name, str) or not isinstance(raw_field, dict):
                raise ValueError("invalid mechanics field")
            field = cast(dict[str, object], raw_field)
            required = {
                "value",
                "unit",
                "status",
                "source_url",
                "source_effective_date",
                "level",
                "inherited_from",
            }
            if not required <= set(field) or set(field) - required - {"evidence_sha256"}:
                raise ValueError("invalid field evidence schema")
            value = field["value"]
            if value is not None and type(value) not in {str, bool, int, float}:
                raise ValueError("invalid mechanics value")
            if not isinstance(field["unit"], str):
                raise ValueError("invalid unit")
            for key in ("source_url", "source_effective_date", "inherited_from"):
                if field[key] is not None and not isinstance(field[key], str):
                    raise ValueError("invalid evidence string")
            evidence = field.get("evidence_sha256")
            if evidence is not None and not isinstance(evidence, str):
                raise ValueError("invalid manual evidence hash")
            fields[name] = MechanicField(
                cast(Value, value),
                field["unit"],
                cast(Status, field["status"]),
                cast(str | None, field["source_url"]),
                cast(str | None, field["source_effective_date"]),
                cast(int | None, field["level"]),
                cast(str | None, field["inherited_from"]),
                evidence,
            )
        entries[token] = MechanicsEntry(
            entry["identity"],
            cast(Literal["card", "tower"], entry["kind"]),
            fields,
            entry["base_identity"],
        )
    return MechanicsCatalog(
        cast(str, body["version"]),
        cast(str, body["era_start"]),
        cast(str, body["era_end"]),
        entries,
        body["synthetic"],
    )


def load(path: str | Path, *, expected_digest: str | None = None) -> MechanicsCatalog:
    catalog = from_payload(loads(Path(path).read_text(encoding="utf-8")))
    from experiments.mechanics.manual_evidence import verify_reports

    verify_reports(catalog, Path(__file__).parent / "evidence")
    if expected_digest is not None and catalog.digest != expected_digest:
        raise ValueError("mechanics digest mismatch")
    return catalog


def synthetic_catalog() -> MechanicsCatalog:
    """Exercise mechanics without claiming invented card statistics as real evidence."""

    def fields(**overrides: bool | int | float | str) -> dict[str, MechanicField]:
        values: dict[str, bool | int | float | str] = {
            name: False for name, unit in FIELD_UNITS.items() if unit == "flag"
        }
        values.update(
            cost_kind="fixed",
            deploy_cost=3,
            ordinary_cycle=True,
            targets_ground=True,
            activation_cycles=0,
            ability_cost=0,
            conditional_cost="none",
        )
        values.update(overrides)
        return {
            name: MechanicField(value, FIELD_UNITS[name], "synthetic")
            for name, value in values.items()
        }

    specs: tuple[dict[str, bool | int | float | str], ...] = (
        {"airborne": True, "building_targeting": True, "deploy_cost": 5},
        {"multi_unit": True, "deploy_cost": 2},
        {"targets_air": True, "area_damage": True, "deploy_cost": 4},
        {"defensive_building": True, "targets_air": True, "deploy_cost": 3},
        {
            "spell": True,
            "spell_damage": True,
            "area_damage": True,
            "targets_air": True,
            "deploy_cost": 2,
        },
        {"spell": True, "spell_control": True, "building_disruption": True, "deploy_cost": 3},
        {"deploy_cost": 1},
        {"deploy_cost": 6},
        {
            "cost_kind": "conditional",
            "conditional_cost": "previous_card_plus_one",
            "ordinary_cycle": False,
            "conditional": True,
        },
        {
            "cost_kind": "conditional",
            "conditional_cost": "ground_3_air_6",
            "ordinary_cycle": True,
            "conditional": True,
        },
    )
    entries = {
        i: MechanicsEntry(f"fixture-{i}:base", "card", fields(**spec))
        for i, spec in enumerate(specs)
    }
    for offset, form in enumerate(("evolution", "hero", "champion"), start=10):
        entries[offset] = MechanicsEntry(
            f"fixture-6:{form}",
            "card",
            fields(
                area_damage=True,
                shield=True,
                control=True,
                spawns_units=True,
                conditional=True,
                activation_cycles=2,
                ability_cost=1,
            ),
            "fixture-6:base",
        )
    for token, name in enumerate(
        ("tower-princess", "cannoneer", "dagger-duchess", "royal-chef"), 13
    ):
        tower_fields = fields(
            targets_air=True, area_damage=False, tower_support=token == 16, burst=token == 15
        )
        for name_field in ("deploy_cost", "cost_kind", "ordinary_cycle", "conditional_cost"):
            tower_fields[name_field] = MechanicField(
                None, FIELD_UNITS[name_field], "not_applicable"
            )
        for name_field, value in (
            ("damage_per_hit", 100 + token),
            ("attack_interval", 1.0),
            ("recharge", 2.0 if token == 15 else 0.0),
        ):
            tower_fields[name_field] = MechanicField(
                value,
                FIELD_UNITS[name_field],
                "synthetic",
                level=16,
            )
        entries[token] = MechanicsEntry(f"{name}:tower", "tower", tower_fields)
    return MechanicsCatalog("synthetic:v1", "2026-09-01", "2026-09-30", entries, True)


def load_partial_catalog() -> MechanicsCatalog:
    return load(Path(__file__).parent / "inputs" / "2026-10-01-partial-r3.json")


def bind_tokens(catalog: MechanicsCatalog, identities: tuple[str, ...]) -> MechanicsCatalog:
    """Bind source identities to the model vocabulary rather than assuming source-ID parity."""
    by_identity = {entry.identity: entry for entry in catalog.entries.values()}
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate model vocabulary identity")
    if any(identity not in by_identity for identity in identities):
        raise ValueError("model vocabulary contains an identity absent from mechanics")
    entries = {token: by_identity[identity] for token, identity in enumerate(identities)}
    return MechanicsCatalog(
        catalog.version, catalog.era_start, catalog.era_end, entries, catalog.synthetic
    )
