"""Manual facts are usable without web URLs and remain bound to their exact report."""

from dataclasses import replace
from hashlib import sha256
from json import dumps
from pathlib import Path

import pytest
from experiments.mechanics.air_audit import air_audit
from experiments.mechanics.contracts import MechanicField
from experiments.mechanics.load import (
    from_payload,
    load,
    load_partial_catalog,
    synthetic_catalog,
    to_payload,
)
from experiments.mechanics.manual_evidence import verify_reports


def test_manual_report_round_trip_binding_and_tamper_detection(tmp_path: Path) -> None:
    catalog = synthetic_catalog()
    entry = catalog.entries[0]
    body = dumps(
        {
            "evidence_version": "manual-mechanics-evidence:v1",
            "source_kind": "user_report",
            "reported_date": "2026-10-01",
            "claims": {entry.identity: {"targets_air": True}},
        }
    ).encode()
    reference = sha256(body).hexdigest()
    path = tmp_path / f"{reference}.json"
    path.write_bytes(body)
    field = MechanicField(
        True, "flag", "user_reported", source_effective_date="2026-10-01", evidence_sha256=reference
    )
    assert field.known and field.source_url is None
    entries = {**catalog.entries, 0: replace(entry, fields={**entry.fields, "targets_air": field})}
    catalog = replace(catalog, entries=entries)
    assert from_payload(to_payload(catalog)).digest == catalog.digest
    verify_reports(catalog, tmp_path)
    changed = replace(
        catalog,
        entries={
            **entries,
            0: replace(entry, fields={**entry.fields, "targets_air": replace(field, value=False)}),
        },
    )
    with pytest.raises(ValueError, match="does not support"):
        verify_reports(changed, tmp_path)
    path.write_bytes(body + b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_reports(catalog, tmp_path)
    path.unlink()
    with pytest.raises(ValueError, match="absent"):
        verify_reports(catalog, tmp_path)


@pytest.mark.parametrize(
    "changes",
    [
        {"evidence_sha256": None},
        {"evidence_sha256": "../escape"},
        {"source_effective_date": None},
        {"source_url": "https://example.test"},
        {"value": None},
    ],
)
def test_manual_facts_cannot_lose_provenance_or_claim_publisher_attribution(
    changes: dict[str, object],
) -> None:
    payload: dict[str, object] = dict(
        value=True,
        unit="flag",
        status="user_reported",
        source_effective_date="2026-10-01",
        evidence_sha256="a" * 64,
    )
    with pytest.raises(ValueError):
        MechanicField(**(payload | changes))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "filename,digest",
    [
        (
            "2026-09-30-partial.json",
            "c505cec2e750e3de226f5a5befe24cba9eab089158ec9b97e1b1c6b4740a64c2",
        ),
        (
            "2026-10-01-partial.json",
            "538cf2d50839fdb0361985f607c612b185303d0ad385b216e56753bf782bcdd0",
        ),
        (
            "2026-10-01-partial-r2.json",
            "1b5bb269c53e2b5a8c866ed50aeb047894015056efdcc47425f1977efa366557",
        ),
    ],
)
def test_old_catalog_hashes_survive_manual_evidence_support(filename: str, digest: str) -> None:
    assert load(Path("experiments/mechanics/inputs") / filename).digest == digest


def test_supplied_targeting_facts_close_all_tower_and_freeze_air_gaps() -> None:
    catalog = load_partial_catalog()
    tokens = [
        t for t, e in catalog.entries.items() if e.kind == "tower" or e.identity == "freeze:base"
    ]
    assert air_audit(catalog, tokens) == {}
    for token in tokens:
        entry = catalog.for_token(token)
        assert entry.flag("targets_air") and entry.flag("targets_ground")
        if entry.identity != "cannoneer:tower":
            assert entry.field("targets_air").status == "user_reported"
    freeze = next(e for e in catalog.entries.values() if e.identity == "freeze:base")
    assert freeze.field("airborne").status == "not_applicable"
