"""Final disposition assignment for one event group after staging.

Adapter and staging records are observations. This module applies duplicate,
conflict, and population precedence to staged rows. Unadaptable rows never
enter an event group.
"""

from collections.abc import Sequence

from clash_sos.domain.canonical import RecordIssue, RecordState
from clash_sos.domain.disposition_ledger import DispositionRow, IngestionState
from clash_sos.domain.staged_dataset import StagedBattleRow, UnadaptableRow


def _sorted_issues(*groups: Sequence[RecordIssue]) -> tuple[RecordIssue, ...]:
    unique = {issue for group in groups for issue in group}
    return tuple(sorted(unique, key=lambda issue: issue.value))


def _adaptable_disposition(
    row: StagedBattleRow,
    *,
    state: IngestionState,
    issues: tuple[RecordIssue, ...],
    representative_archive_member: str | None,
    representative_row_number: int | None,
) -> DispositionRow:
    return DispositionRow(
        archive_member=row.archive_member,
        row_number=row.row_number,
        state=state,
        issues=issues,
        source_id=row.source_id,
        timestamp=row.timestamp,
        mode=row.mode,
        balance_era_id=row.balance_era_id,
        outcome=row.outcome,
        event_key=row.event_key,
        fingerprint=row.fingerprint,
        side_a_player_id=row.side_a_player_id,
        side_b_player_id=row.side_b_player_id,
        representative_archive_member=representative_archive_member,
        representative_row_number=representative_row_number,
    )


def disposition_from_unadaptable(row: UnadaptableRow) -> DispositionRow:
    """Copy an unadaptable staging row into the unified ledger without identity fields."""
    return DispositionRow(
        archive_member=row.archive_member,
        row_number=row.row_number,
        state=row.state,
        issues=row.issues,
    )


def finalize_event_group(rows: Sequence[StagedBattleRow]) -> tuple[DispositionRow, ...]:
    """Assign final states for one event_key group ordered by source location."""
    ordered = tuple(sorted(rows, key=lambda row: (row.archive_member, row.row_number)))
    if not ordered:
        return ()
    fingerprints = {row.fingerprint for row in ordered}
    if len(fingerprints) > 1:
        return tuple(
            _adaptable_disposition(
                row,
                state=RecordState.QUARANTINED,
                issues=_sorted_issues((RecordIssue.CONFLICTING_BATTLE,), row.observation_issues),
                representative_archive_member=None,
                representative_row_number=None,
            )
            for row in ordered
        )

    representative = ordered[0]
    representative_member = representative.archive_member
    representative_row_number = representative.row_number
    if representative.observation_issues:
        lead = _adaptable_disposition(
            representative,
            state=RecordState.UNSUPPORTED,
            issues=representative.observation_issues,
            representative_archive_member=representative_member,
            representative_row_number=representative_row_number,
        )
    else:
        lead = _adaptable_disposition(
            representative,
            state=RecordState.VALID,
            issues=(),
            representative_archive_member=representative_member,
            representative_row_number=representative_row_number,
        )
    duplicates = tuple(
        _adaptable_disposition(
            row,
            state=RecordState.QUARANTINED,
            issues=_sorted_issues((RecordIssue.DUPLICATE_BATTLE,), row.observation_issues),
            representative_archive_member=representative_member,
            representative_row_number=representative_row_number,
        )
        for row in ordered[1:]
    )
    return (lead, *duplicates)
