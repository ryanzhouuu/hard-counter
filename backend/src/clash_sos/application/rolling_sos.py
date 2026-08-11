"""Rolling strength-of-schedule analytics over canonical battle records."""

from collections import Counter
from collections.abc import Iterable

from clash_sos.domain.analytics import (
    AnalysisState,
    BattleAnalysisRecord,
    ExclusionReasonCount,
    PredictionState,
    RollingSoSObservation,
    RollingSoSResult,
)
from clash_sos.domain.canonical import BattleOutcome, PlayerId, RecordIssue, RecordState

SUPPORTED_MODE = "Ranked1v1_NewArena"
MAX_CARD_LEVEL = 16


class RollingSoSAnalyzer:
    """Calculate the approved final-N eligible battle window for one player."""

    def analyze(
        self,
        target_player: PlayerId | str,
        records: Iterable[BattleAnalysisRecord],
        window_size: int,
    ) -> RollingSoSResult:
        if window_size <= 0:
            raise ValueError("window_size must be positive")

        player = target_player if isinstance(target_player, PlayerId) else PlayerId(target_player)
        eligible: list[RollingSoSObservation] = []
        exclusion_counts: Counter[RecordIssue] = Counter()
        record_count = 0

        for record in records:
            record_count += 1
            issues = self._exclusion_issues(record, player)
            if issues:
                exclusion_counts.update(issues)
                continue
            eligible.append(self._orient(record, player))

        eligible.sort(
            key=lambda observation: (observation.timestamp, observation.battle_fingerprint)
        )
        exclusion_reasons = tuple(
            ExclusionReasonCount(issue=issue, count=count)
            for issue, count in sorted(exclusion_counts.items(), key=lambda item: item[0].value)
        )
        excluded_count = record_count - len(eligible)
        available_provenance = tuple(
            dict.fromkeys(observation.provenance for observation in eligible)
        )

        if len(eligible) < window_size:
            return RollingSoSResult(
                status=AnalysisState.INSUFFICIENT_DATA,
                target_player=player,
                requested_window=window_size,
                eligible_count=len(eligible),
                excluded_count=excluded_count,
                exclusion_reasons=exclusion_reasons,
                provenance=available_provenance,
            )

        window = tuple(eligible[-window_size:])
        expected_wins = sum(observation.player_win_probability for observation in window)
        actual_wins = sum(observation.actual_win for observation in window)
        strength_of_schedule = sum(observation.difficulty for observation in window) / window_size
        provenance = tuple(dict.fromkeys(observation.provenance for observation in window))

        return RollingSoSResult(
            status=AnalysisState.AVAILABLE,
            target_player=player,
            requested_window=window_size,
            eligible_count=len(eligible),
            excluded_count=excluded_count,
            exclusion_reasons=exclusion_reasons,
            window=window,
            strength_of_schedule=strength_of_schedule,
            expected_wins=expected_wins,
            actual_wins=actual_wins,
            performance_above_expectation=actual_wins - expected_wins,
            provenance=provenance,
        )

    def _exclusion_issues(
        self, record: BattleAnalysisRecord, target_player: PlayerId
    ) -> tuple[RecordIssue, ...]:
        if record.disposition.state is not RecordState.VALID:
            return record.disposition.issues

        battle = record.battle
        issues: list[RecordIssue] = []
        if battle.mode != SUPPORTED_MODE:
            issues.append(RecordIssue.UNSUPPORTED_MODE)
        all_levels = battle.side_a.card_levels + battle.side_b.card_levels
        if any(level != MAX_CARD_LEVEL for level in all_levels):
            issues.append(RecordIssue.NON_MAX_CARD_LEVEL)
        if battle.outcome is BattleOutcome.DRAW:
            issues.append(RecordIssue.DRAW_OUTCOME)
        if target_player not in (battle.side_a.player_id, battle.side_b.player_id):
            issues.append(RecordIssue.TARGET_PLAYER_NOT_FOUND)
        if issues:
            return tuple(issues)

        prediction = record.prediction
        if prediction is None:
            return (RecordIssue.UNAVAILABLE_MODEL_COVERAGE,)
        if prediction.state is PredictionState.UNAVAILABLE:
            return (prediction.issue or RecordIssue.UNAVAILABLE_MODEL_COVERAGE,)
        if prediction.provenance is None:
            return (RecordIssue.UNAVAILABLE_MODEL_COVERAGE,)
        if (
            prediction.provenance.balance_era_id != battle.balance_era.era_id
            or prediction.provenance.card_catalog_version != battle.balance_era.card_catalog_version
        ):
            return (RecordIssue.STALE_BALANCE_ERA,)
        return ()

    def _orient(
        self, record: BattleAnalysisRecord, target_player: PlayerId
    ) -> RollingSoSObservation:
        battle = record.battle
        prediction = record.prediction
        if (
            prediction is None
            or prediction.side_a_win_probability is None
            or prediction.provenance is None
        ):
            raise ValueError("eligible records require an available prediction")

        target_is_side_a = target_player == battle.side_a.player_id
        player_win_probability = (
            prediction.side_a_win_probability
            if target_is_side_a
            else 1 - prediction.side_a_win_probability
        )
        actual_win = int(
            battle.outcome
            is (BattleOutcome.SIDE_A_WIN if target_is_side_a else BattleOutcome.SIDE_B_WIN)
        )
        return RollingSoSObservation(
            timestamp=battle.timestamp,
            battle_fingerprint=battle.fingerprint,
            player_win_probability=player_win_probability,
            actual_win=actual_win,
            difficulty=1 - player_win_probability,
            provenance=prediction.provenance,
        )


def calculate_rolling_sos(
    target_player: PlayerId | str,
    records: Iterable[BattleAnalysisRecord],
    window_size: int,
) -> RollingSoSResult:
    """Calculate rolling SoS through the application-layer analyzer."""

    return RollingSoSAnalyzer().analyze(target_player, records, window_size)
