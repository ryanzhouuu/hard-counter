from dataclasses import replace
from datetime import UTC, datetime

import pytest
from experiments.common.windows import compare_windows, summarize_windows
from experiments.tests.common.reporting_fixtures import prediction, provenance

from clash_sos.domain.analytics import AnalysisState


def test_final_windows_orient_players_and_preserve_incomplete_counts() -> None:
    baseline = tuple(prediction(i, 0.5, player_b="B" if i < 10 else "C") for i in range(12))
    candidate = tuple(replace(row, probability=0.7) for row in reversed(baseline))
    windows = compare_windows(
        candidate,
        baseline,
        candidate_provenance=provenance("candidate"),
        comparator_provenance=provenance("baseline"),
    )
    indexed = {(row.player, row.window_size): row for row in windows}
    a, b, c = (indexed[(player, 10)] for player in ("A", "B", "C"))
    assert a.candidate.actual_wins == 10
    assert a.candidate.expected_wins == pytest.approx(7)
    assert a.expected_win_shift == pytest.approx(2)
    assert a.difficulty_shift == pytest.approx(-0.2)
    assert b.candidate.actual_wins == 0
    assert b.candidate.expected_wins == pytest.approx(3)
    assert b.expected_win_shift == pytest.approx(-2)
    assert c.candidate.status is AnalysisState.INSUFFICIENT_DATA
    assert c.candidate.eligible_count == 2
    assert c.expected_win_shift is None
    assert indexed[("A", 25)].candidate.status is AnalysisState.INSUFFICIENT_DATA
    summaries = summarize_windows(windows)
    assert summaries[0].complete_count == 2
    assert summaries[0].incomplete_count == 1
    assert summaries[0].mean_signed_expected_win_shift == pytest.approx(0)
    assert summaries[0].p50_absolute_expected_win_shift == pytest.approx(2)
    assert summaries[0].p90_absolute_expected_win_shift == pytest.approx(2)
    assert summaries[1].complete_count == 0
    assert summaries[1].p90_absolute_expected_win_shift is None
    assert tuple(row.battle_fingerprint for row in a.candidate.window) == tuple(
        f"f{index:04}" for index in range(2, 12)
    )


def test_timestamps_then_fingerprint_determine_final_window() -> None:
    stamp = datetime(2026, 1, 1, tzinfo=UTC)
    rows = tuple(prediction(index, timestamp=stamp) for index in (2, 0, 1))
    windows = compare_windows(
        rows,
        rows,
        candidate_provenance=provenance("candidate"),
        comparator_provenance=provenance("baseline"),
        window_sizes=(2,),
    )
    assert tuple(item.battle_fingerprint for item in windows[0].candidate.window) == (
        "f0001",
        "f0002",
    )
    assert windows[0].expected_win_shift == 0
    assert windows[0].candidate.performance_above_expectation == 1


def test_window_population_mismatch_and_invalid_window_sizes_fail() -> None:
    rows = (prediction(0), prediction(1))
    with pytest.raises(ValueError, match="identical canonical"):
        compare_windows(
            rows,
            rows[:1],
            candidate_provenance=provenance("candidate"),
            comparator_provenance=provenance("baseline"),
        )
    for sizes in ((), (0,), (10, 10)):
        with pytest.raises(ValueError, match="window sizes"):
            compare_windows(
                rows,
                rows,
                candidate_provenance=provenance("candidate"),
                comparator_provenance=provenance("baseline"),
                window_sizes=sizes,
            )


def test_seed_dispersion_uses_identical_windows_and_visible_incompletion() -> None:
    from experiments.common.windows import window_seed_variability

    baseline = tuple(prediction(index) for index in range(10))
    seeds = tuple(
        compare_windows(
            tuple(replace(row, probability=probability) for row in baseline),
            baseline,
            candidate_provenance=provenance("candidate"),
            comparator_provenance=provenance("baseline"),
        )
        for probability in (0.6, 0.8)
    )
    result = {(row.player, row.window_size): row for row in window_seed_variability(seeds)}
    assert result[("A", 10)].expected_win_shift_standard_deviation == pytest.approx(1)
    assert result[("A", 10)].expected_win_shift_span == pytest.approx(2)
    assert result[("A", 25)].expected_win_shift_standard_deviation is None
    with pytest.raises(ValueError):
        window_seed_variability((seeds[0], seeds[1][:-1]))
    with pytest.raises(ValueError):
        window_seed_variability((seeds[0], (*seeds[1], seeds[1][0])))
    with pytest.raises(ValueError):
        window_seed_variability(())
