"""Research mechanics preserve missingness and field-level source evidence."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from json import dumps
from math import isfinite
from types import MappingProxyType
from typing import Literal, cast

from experiments.mechanics.fields import CONDITIONAL_CATEGORIES
from experiments.mechanics.fields import FIELD_UNITS as FIELD_UNITS
from experiments.mechanics.fields import Status as Status
from experiments.mechanics.fields import Value as Value


class MechanicsUnavailable(ValueError):
    """A branch requires evidence absent from the frozen catalog."""


@dataclass(frozen=True)
class MechanicField:
    value: Value
    unit: str
    status: Status
    source_url: str | None = None
    source_effective_date: str | None = None
    level: int | None = None
    inherited_from: str | None = None
    evidence_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {
            "verified",
            "user_reported",
            "unknown",
            "not_applicable",
            "synthetic",
        }:
            raise ValueError("invalid verification status")
        if self.status in {"unknown", "not_applicable"} and self.value is not None:
            raise ValueError("unknown and inapplicable fields have no value")
        if self.known and self.value is None:
            raise ValueError("known fields require a value")
        if isinstance(self.value, float) and not isfinite(self.value):
            raise ValueError("mechanics must be finite")
        if self.source_effective_date is not None:
            date.fromisoformat(self.source_effective_date)
        if self.source_url is not None and not self.source_url.startswith("https://"):
            raise ValueError("mechanics evidence requires an HTTPS source URL")
        if self.status == "verified":
            if not self.source_url or not self.source_url.startswith("https://"):
                raise ValueError("verified mechanics require a source URL")
            if not self.source_effective_date:
                raise ValueError("verified mechanics require a source date")
            date.fromisoformat(self.source_effective_date)
        if self.status == "user_reported":
            if self.source_url is not None or not self.source_effective_date:
                raise ValueError("user reports require a date and no publisher URL")
            if (
                not self.evidence_sha256
                or len(self.evidence_sha256) != 64
                or any(char not in "0123456789abcdef" for char in self.evidence_sha256)
            ):
                raise ValueError("user reports require a saved evidence hash")
        elif self.evidence_sha256 is not None:
            raise ValueError("manual evidence hashes belong to user reports")
        if self.level is not None and (type(self.level) is not int or self.level != 16):
            raise ValueError("quantitative mechanics require normalized level 16")

    @property
    def known(self) -> bool:
        return self.status in {"verified", "user_reported", "synthetic"}


@dataclass(frozen=True)
class MechanicsEntry:
    identity: str
    kind: Literal["card", "tower"]
    fields: Mapping[str, MechanicField]
    base_identity: str | None = None

    def __post_init__(self) -> None:
        if not self.identity.strip() or self.kind not in {"card", "tower"}:
            raise ValueError("identity and card/tower kind are required")
        for name, field in self.fields.items():
            if name not in FIELD_UNITS or field.unit != FIELD_UNITS[name]:
                raise ValueError(f"invalid mechanics field/unit: {name}")
            if field.known and field.unit == "flag" and type(field.value) is not bool:
                raise ValueError(f"{name} requires a boolean")
            if field.known and field.unit in {"elixir", "deployments", "damage", "seconds"}:
                if (
                    type(field.value) not in {int, float}
                    or float(cast(int | float, field.value)) < 0
                ):
                    raise ValueError(f"{name} requires a nonnegative number")
                if name == "attack_interval" and float(cast(int | float, field.value)) == 0:
                    raise ValueError("attack interval must be positive")
            if field.known and field.unit == "category" and not isinstance(field.value, str):
                raise ValueError(f"{name} requires a category")
            if name == "cost_kind" and field.known and field.value not in {"fixed", "conditional"}:
                raise ValueError("unregistered deployment cost kind")
            if (
                name == "airborne_mode"
                and field.known
                and field.value not in {"ground", "air", "hybrid"}
            ):
                raise ValueError("unregistered airborne mode")
            if name == "activation_cycles" and field.known and type(field.value) is not int:
                raise ValueError("activation cycles require whole deployments")
            if (
                name in CONDITIONAL_CATEGORIES
                and field.known
                and field.value not in CONDITIONAL_CATEGORIES[name]
            ):
                raise ValueError(f"unregistered conditional category: {name}")
            if field.inherited_from and not field.known:
                raise ValueError("field inheritance requires verified applicability")
            if name == "damage_per_hit" and field.known and field.level != 16:
                raise ValueError("damage requires normalized level 16")
        if self.kind == "tower" and any(
            self.fields.get(name, unknown(name)).status != "not_applicable"
            for name in ("deploy_cost", "cost_kind", "ordinary_cycle")
        ):
            raise ValueError("tower deployment and cycle fields must be inapplicable")
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))

    def field(self, name: str) -> MechanicField:
        return self.fields.get(name, unknown(name))

    def flag(self, name: str) -> bool:
        field = self.field(name)
        return field.known and field.value is True

    def number(self, name: str) -> float | None:
        field = self.field(name)
        return (
            float(cast(int | float, field.value))
            if field.known and type(field.value) in {int, float}
            else None
        )


@dataclass(frozen=True)
class MechanicsCatalog:
    version: str
    era_start: str
    era_end: str
    entries: Mapping[int, MechanicsEntry]
    synthetic: bool = False

    def __post_init__(self) -> None:
        if not self.version.strip() or not self.entries:
            raise ValueError("mechanics version and entries are required")
        if date.fromisoformat(self.era_start) > date.fromisoformat(self.era_end):
            raise ValueError("invalid mechanics era")
        identities = [entry.identity for entry in self.entries.values()]
        if len(set(identities)) != len(identities):
            raise ValueError("duplicate mechanics identity")
        if any(type(token) is not int or token < 0 for token in self.entries):
            raise ValueError("mechanics tokens must be nonnegative integers")
        for entry in self.entries.values():
            if entry.base_identity and entry.base_identity not in identities:
                raise ValueError("form base identity is absent")
            for field in entry.fields.values():
                if field.status == "synthetic" and not self.synthetic:
                    raise ValueError("synthetic evidence cannot enable real mechanics")
                if field.inherited_from and field.inherited_from != entry.base_identity:
                    raise ValueError("inheritance must reference the declared base")
        object.__setattr__(self, "entries", MappingProxyType(dict(self.entries)))

    def for_token(self, token: int, *, inherited: bool = False) -> MechanicsEntry:
        entry = self.entries.get(token)
        if entry is None:
            raise MechanicsUnavailable(f"mechanics absent for token {token}")
        if inherited and entry.base_identity:
            return next(e for e in self.entries.values() if e.identity == entry.base_identity)
        fields = dict(entry.fields)
        for name, field in entry.fields.items():
            if field.inherited_from:
                base = next(e for e in self.entries.values() if e.identity == field.inherited_from)
                source = base.field(name)
                if not source.known:
                    raise MechanicsUnavailable(
                        f"unverified inherited field {entry.identity}/{name}"
                    )
                fields[name] = MechanicField(
                    source.value,
                    source.unit,
                    field.status,
                    field.source_url,
                    field.source_effective_date,
                    source.level,
                    field.inherited_from,
                    field.evidence_sha256,
                )
        return MechanicsEntry(entry.identity, entry.kind, fields, entry.base_identity)

    @property
    def digest(self) -> str:
        from experiments.mechanics.load import to_payload

        return sha256(
            dumps(to_payload(self), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def coverage(self, tokens: Sequence[int], fields: Sequence[str]) -> dict[str, dict[str, str]]:
        return {
            self.for_token(token).identity: {
                name: self.for_token(token).field(name).status for name in fields
            }
            for token in sorted(set(tokens))
        }


def unknown(name: str) -> MechanicField:
    return MechanicField(None, FIELD_UNITS[name], "unknown")


@dataclass(frozen=True)
class FeatureResult:
    names: tuple[str, ...]
    values: tuple[float, ...]
    formulas: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.names) != len(self.values) or len(self.names) != len(self.formulas):
            raise ValueError("feature registry lengths differ")
        if len(set(self.names)) != len(self.names) or not all(isfinite(v) for v in self.values):
            raise ValueError("features require unique names and finite values")


def decode_pair(
    tokens: Sequence[Sequence[int]],
    catalog: MechanicsCatalog,
    *,
    inherited: bool = False,
) -> tuple[tuple[MechanicsEntry, ...], tuple[MechanicsEntry, ...]]:
    """Require eight deployment slots followed by one static tower on each side."""
    if len(tokens) != 2 or any(len(side) != 9 for side in tokens):
        raise ValueError("tokens must have shape [2, 9]")
    sides: list[tuple[MechanicsEntry, ...]] = []
    for side in tokens:
        entries = tuple(catalog.for_token(token, inherited=inherited) for token in side)
        if any(entry.kind != "card" for entry in entries[:8]) or entries[8].kind != "tower":
            raise ValueError("tower mechanics belong only in the ninth slot")
        sides.append(entries)
    return sides[0], sides[1]


def difference(
    names: Sequence[str],
    left: Sequence[float],
    right: Sequence[float],
    formulas: Sequence[str],
) -> FeatureResult:
    return FeatureResult(
        tuple(names), tuple(a - b for a, b in zip(left, right, strict=True)), tuple(formulas)
    )
