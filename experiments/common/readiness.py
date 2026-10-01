"""Covariate and provenance readiness without outcome scoring or advancement thresholds."""

from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import date
from hashlib import sha256

from clash_sos.domain.attention_schema import AttentionCardSchema
from clash_sos.domain.canonical_dataset import canonical_json_bytes
from experiments.common.data_access import ResearchRow, RoleAccess
from experiments.common.protocol import require_search
from experiments.higher_order.features import extract as pattern_features
from experiments.higher_order.patterns import PATTERNS
from experiments.mechanics.contracts import FIELD_UNITS, MechanicsCatalog, MechanicsUnavailable


def _lineup(tokens: tuple[int, ...]) -> str:
    return sha256(canonical_json_bytes(tokens)).hexdigest()


def _covariates(rows: Sequence[ResearchRow], catalog: MechanicsCatalog) -> dict[str, object]:
    players: Counter[str] = Counter()
    towers: Counter[str] = Counter()
    forms: Counter[str] = Counter()
    lineups: Counter[str] = Counter()
    pairs: Counter[str] = Counter()
    player_lineups: defaultdict[str, set[str]] = defaultdict(set)
    first_seen: dict[str, str] = {}
    for row in rows:
        a, b = (_lineup(side) for side in row.tokens)
        pairs["|".join(sorted((a, b)))] += 1
        for player, tokens, lineup in zip(
            (row.player_a, row.player_b), row.tokens, (a, b), strict=True
        ):
            players[player] += 1
            lineups[lineup] += 1
            player_lineups[player].add(lineup)
            first_seen.setdefault(player, row.key[0].date().isoformat())
            for token in tokens:
                entry = catalog.entries.get(token)
                if entry is None:
                    continue
                if entry.kind == "tower":
                    towers[entry.identity] += 1
                elif not entry.identity.endswith(":base"):
                    forms[entry.identity] += 1
    return {
        "row_count": len(rows),
        "distinct_timestamps": len({row.key[0] for row in rows}),
        "first_timestamp": min(row.key[0] for row in rows).isoformat() if rows else None,
        "last_timestamp": max(row.key[0] for row in rows).isoformat() if rows else None,
        "daily_rows": dict(sorted(Counter(row.key[0].date().isoformat() for row in rows).items())),
        "player_count": len(players),
        "repeated_player_count": sum(count > 1 for count in players.values()),
        "players_with_multiple_lineups": sum(len(values) > 1 for values in player_lineups.values()),
        "max_player_match_fraction": max(players.values()) / len(rows) if rows else None,
        "player_first_seen_days": dict(sorted(Counter(first_seen.values()).items())),
        "lineup_support": dict(sorted(lineups.items())),
        "pair_support": dict(sorted(pairs.items())),
        "tower_support": dict(sorted(towers.items())),
        "form_support": dict(sorted(forms.items())),
        "complete_player_windows": {
            str(size): sum(count >= size for count in players.values()) for size in (10, 25)
        },
    }


def readiness(
    access: RoleAccess,
    catalog: MechanicsCatalog,
    schema: AttentionCardSchema | None = None,
) -> dict[str, object]:
    require_search(access.protocol)
    populations = {
        "selection_fit": access.read("selection_fit", "fit"),
        "watch": access.read("watch", "fit"),
        "refit": access.read("refit", "fit"),
        "calibration": access.read("calibration", "calibrate"),
        "development": access.read("development", "compare"),
    }
    rows = tuple(
        sorted(
            (row for role in ("refit", "calibration", "development") for row in populations[role]),
            key=lambda row: row.key,
        )
    )
    tokens = sorted({token for row in rows for side in row.tokens for token in side})
    coverage: dict[str, dict[str, str]] = {}
    issues: set[str] = set()
    for token in tokens:
        entry = catalog.entries.get(token)
        if entry is None:
            issues.add(f"mechanics absent for token {token}")
            coverage[f"token:{token}"] = {name: "unknown" for name in FIELD_UNITS}
            continue
        try:
            resolved = catalog.for_token(token)
        except MechanicsUnavailable as error:
            issues.add(str(error))
            coverage[entry.identity] = {name: "unknown" for name in FIELD_UNITS}
            continue
        coverage[entry.identity] = {name: resolved.field(name).status for name in FIELD_UNITS}
        if schema is not None and (
            token >= len(schema.identity_vocab) or schema.identity_vocab[token] != entry.identity
        ):
            issues.add(f"mechanics/schema identity mismatch at token {token}")
    if not catalog.synthetic and any(
        not date.fromisoformat(catalog.era_start)
        <= row.key[0].date()
        <= date.fromisoformat(catalog.era_end)
        for row in rows
    ):
        issues.add("population battle dates exceed verified mechanics era")
    if schema is not None and schema.balance_era_id != access.protocol.balance_era_id:
        issues.add("schema and population era mismatch")
    support = {pattern.name: 0 for pattern in PATTERNS}
    pattern_issues: set[str] = set()
    for row in populations["refit"]:
        try:
            features = pattern_features(row.tokens, catalog)
        except MechanicsUnavailable as error:
            pattern_issues.add(str(error))
            continue
        for pattern, value in zip(PATTERNS, features.values, strict=True):
            support[pattern.name] += int(value != 0)
    unknown = sum(status == "unknown" for fields in coverage.values() for status in fields.values())
    return {
        "population": _covariates(rows, catalog),
        "roles": {
            role: _covariates(population, catalog) for role, population in populations.items()
        },
        "mechanics": {
            "version": catalog.version,
            "sha256": catalog.digest,
            "synthetic": catalog.synthetic,
            "era_start": catalog.era_start,
            "era_end": catalog.era_end,
            "coverage": coverage,
            "unknown_field_count": unknown,
            "compatibility_issues": sorted(issues),
        },
        "patterns": {
            "refit_support": support,
            "status": "unavailable" if pattern_issues or issues else "available",
            "disabled_reasons": sorted(pattern_issues | issues),
        },
    }
