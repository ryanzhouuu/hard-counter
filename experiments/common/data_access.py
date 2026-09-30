from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from clash_sos.domain.attention_protocol import AttentionProtocol, RowKey, digest_row_keys
from experiments.common.contracts import PopulationIdentity, Role
from experiments.common.protocol import require_search

Operation = Literal["fit", "calibrate", "compare"]
_ALLOWED: dict[Operation, frozenset[Role]] = {
    "fit": frozenset(("selection_fit", "watch", "refit")),
    "calibrate": frozenset(("calibration",)),
    "compare": frozenset(("development",)),
}


@dataclass(frozen=True)
class ResearchRow:
    key: RowKey
    event_key: str
    player_a: str
    player_b: str
    label: int
    tokens: tuple[tuple[int, ...], tuple[int, ...]]
    mirrored: bool = False

    def __post_init__(self) -> None:
        if self.key[0].tzinfo is None or self.key[0].utcoffset() is None:
            raise ValueError("row timestamp must be aware")
        if not self.event_key or not self.player_a or not self.player_b:
            raise ValueError("event and players must be present")
        if self.player_a == self.player_b or self.label not in (0, 1):
            raise ValueError("distinct players and decisive label required")
        if len(self.tokens[0]) not in (8, 9) or len(self.tokens[0]) != len(self.tokens[1]):
            raise ValueError("both sides need eight cards and optionally one tower")
        if any(len(set(side)) != len(side) or min(side) < 0 for side in self.tokens):
            raise ValueError("side identities must be distinct nonnegative tokens")

    def swapped(self) -> "ResearchRow":
        return ResearchRow(
            self.key,
            self.event_key,
            self.player_b,
            self.player_a,
            1 - self.label,
            (self.tokens[1], self.tokens[0]),
            not self.mirrored,
        )


def validate_rows(rows: Sequence[ResearchRow]) -> None:
    if not rows or len({r.event_key for r in rows}) != len(rows):
        raise ValueError("row population must be nonempty with one row per event")
    digest_row_keys(r.key for r in rows)


class RoleAccess:
    def __init__(self, protocol: AttentionProtocol, rows: Sequence[ResearchRow]) -> None:
        require_search(protocol)
        validate_rows(rows)
        self.protocol = protocol
        self._roles: dict[Role, tuple[ResearchRow, ...]] = {}
        for role in ("selection_fit", "watch", "refit", "calibration", "development"):
            part = getattr(protocol, role)
            if part is None:
                raise ValueError(f"missing {role}")
            selected = tuple(r for r in rows if part.start <= r.key[0] < part.end)
            if len(selected) != part.row_count or digest_row_keys(r.key for r in selected) != (
                part.row_keys_sha256
            ):
                raise ValueError(f"{role} row count or digest mismatch")
            self._roles[role] = selected
        if set(r.key for r in rows) != set(
            r.key for role in ("refit", "calibration", "development") for r in self._roles[role]
        ):
            raise ValueError("research inventory contains undeclared rows")

    def read(self, role: Role, operation: Operation) -> tuple[ResearchRow, ...]:
        if role not in _ALLOWED[operation]:
            raise PermissionError(f"{operation} cannot read {role}")
        return self._roles[role]


@dataclass(frozen=True)
class ReportingContract:
    population: PopulationIdentity
    fit_population: PopulationIdentity
    candidate_sha256: str
    inspected_through: datetime
    frozen_at: datetime

    def __post_init__(self) -> None:
        future, fit = self.population, self.fit_population
        for stamp in (self.inspected_through, self.frozen_at):
            if stamp.tzinfo is None or stamp.utcoffset() is None:
                raise ValueError("reporting cutoffs must be aware")
        if future.start <= max(fit.end, self.inspected_through, self.frozen_at):
            raise ValueError("reporting must be strictly future by battle time")
        if (future.era, future.mode, future.level, future.catalog_sha256) != (
            fit.era,
            fit.mode,
            fit.level,
            fit.catalog_sha256,
        ):
            raise ValueError("reporting population is incompatible with the frozen study")
        if {f.sha256 for f in future.snapshot_files} & {f.sha256 for f in fit.snapshot_files}:
            raise ValueError("reporting files must be physically separate")
        if len(self.candidate_sha256) != 64:
            raise ValueError("candidate digest required")


class ReportAccess:
    def __init__(self, contract: ReportingContract, rows: Sequence[ResearchRow]) -> None:
        validate_rows(rows)
        population = contract.population
        if any(not population.start <= r.key[0] < population.end for r in rows):
            raise ValueError("reporting rows must respect battle-time cutoffs")
        if digest_row_keys(r.key for r in rows) != population.row_keys_sha256:
            raise ValueError("reporting row inventory mismatch")
        self.contract = contract
        self._rows = tuple(rows)

    def read(
        self, role: Literal["reporting"], operation: Literal["report"]
    ) -> tuple[ResearchRow, ...]:
        if role != "reporting" or operation != "report":
            raise PermissionError("reporting cannot fit, calibrate, or search")
        return self._rows
