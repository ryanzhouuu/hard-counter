"""Bind manual mechanics to saved user statements instead of invented web citations."""

from hashlib import sha256
from json import loads
from pathlib import Path
from typing import cast

from experiments.mechanics.contracts import MechanicsCatalog


def verify_reports(catalog: MechanicsCatalog, directory: Path) -> None:
    """Check evidence bytes, report date, identity, and each manually supplied value."""
    reports: dict[str, object] = {}
    for entry in catalog.entries.values():
        for name, field in entry.fields.items():
            if field.status != "user_reported":
                continue
            reference = field.evidence_sha256
            assert reference is not None
            if reference not in reports:
                path = directory / f"{reference}.json"
                if not path.is_file():
                    raise ValueError("saved manual evidence is absent")
                raw = path.read_bytes()
                if sha256(raw).hexdigest() != reference:
                    raise ValueError("manual evidence hash mismatch")
                reports[reference] = loads(raw)
            report = reports[reference]
            if not isinstance(report, dict):
                raise ValueError("invalid manual evidence metadata")
            report = cast(dict[str, object], report)
            if (
                report.get("evidence_version") != "manual-mechanics-evidence:v1"
                or report.get("source_kind") != "user_report"
                or report.get("reported_date") != field.source_effective_date
            ):
                raise ValueError("invalid manual evidence metadata")
            claims = report.get("claims")
            if not isinstance(claims, dict):
                raise ValueError("invalid manual evidence claims")
            claims = cast(dict[str, object], claims)
            values = claims.get(entry.identity)
            if not isinstance(values, dict):
                raise ValueError("manual evidence does not support this field")
            values = cast(dict[str, object], values)
            if (
                name not in values
                or type(values[name]) is not type(field.value)
                or values[name] != field.value
            ):
                raise ValueError("manual evidence does not support this field")
