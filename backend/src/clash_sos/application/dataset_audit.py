"""Generic identity checks used while auditing source datasets."""

from clash_sos.domain.canonical import (
    Battle,
    RecordDisposition,
    RecordIssue,
    RecordState,
)


class BattleIdentityIndex:
    def __init__(self) -> None:
        self._events: dict[str, tuple[str, str]] = {}

    def observe(self, battle: Battle, *, location: str) -> RecordDisposition | None:
        first = self._events.get(battle.event_key)
        if first is None:
            self._events[battle.event_key] = (battle.fingerprint, location)
            return None

        fingerprint, first_location = first
        issue = (
            RecordIssue.DUPLICATE_BATTLE
            if fingerprint == battle.fingerprint
            else RecordIssue.CONFLICTING_BATTLE
        )
        return RecordDisposition(
            state=RecordState.QUARANTINED,
            issues=(issue,),
            detail=f"first observed at {first_location}",
        )
