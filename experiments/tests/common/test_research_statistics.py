from dataclasses import replace
from math import log

import pytest
from experiments.common.statistics import dyadic_interval, paired_comparison, score_predictions
from experiments.tests.common.reporting_fixtures import prediction


def test_production_metrics_against_independent_calculations() -> None:
    rows = (prediction(0, 0.5), prediction(1, 0.8), prediction(2, 0.3, label=0))
    result = score_predictions(rows)
    assert result.metrics.log_loss == pytest.approx(-(log(0.5) + log(0.8) + log(0.7)) / 3)
    assert result.metrics.brier_score == pytest.approx((0.25 + 0.04 + 0.09) / 3)
    assert result.accuracy_half_ties == pytest.approx(2.5 / 3)
    assert result.metrics.expected_calibration_error == pytest.approx((0.5 + 0.2 + 0.3) / 3)
    assert sum(item.count for item in result.reliability) == 3
    assert result.reliability[0].mean_prediction is None


@pytest.mark.parametrize("offset", range(8))
def test_direct_dyadic_sum_and_reverse_rematches(offset: int) -> None:
    first = ("A", "B", "B", "D", "A", "G", "C")
    second = ("B", "A", "C", "E", "F", "H", "D")
    delta = tuple(float(((index + offset) ** 2) % 13) for index in range(len(first)))
    mean = sum(delta) / len(delta)
    direct = (
        sum(
            (delta[i] - mean) * (delta[j] - mean)
            for i in range(len(delta))
            for j in range(len(delta))
            if {first[i], second[i]} & {first[j], second[j]}
        )
        / len(delta) ** 2
    )
    result = dyadic_interval(delta, first, second)
    assert result.variance == pytest.approx(direct, abs=1e-14)
    assert result.unordered_pair_count == 6
    assert result == dyadic_interval(delta, second, first)
    assert result.max_player_match_fraction == pytest.approx(3 / 7)


def test_negative_and_zero_variance_remain_visible() -> None:
    result = dyadic_interval((1, -1, 1, -1), ("A", "B", "C", "D"), ("B", "C", "D", "A"))
    assert result.variance < 0
    assert result.status == "negative_variance_estimate"
    assert result.standard_error is None
    assert result.ci95 is None
    zero = dyadic_interval((1, 1), ("A", "C"), ("B", "D"))
    assert zero.status == "zero_variance_estimate"
    assert zero.small_player_count


def test_disjoint_dyads_reduce_to_uncorrected_independent_variance() -> None:
    delta = (0.1, 0.4, -0.2, 0.3)
    mean = sum(delta) / 4
    result = dyadic_interval(delta, ("A", "B", "C", "D"), ("E", "F", "G", "H"))
    assert result.variance == pytest.approx(sum((value - mean) ** 2 for value in delta) / 16)
    for differences, first, second in (
        ((1,), ("A",), ("B",)),
        ((1, 2), ("A",), ("B",)),
        ((1, 2), ("A", "B"), ("A", "C")),
        ((1, float("nan")), ("A", "B"), ("B", "C")),
    ):
        with pytest.raises(ValueError):
            dyadic_interval(differences, first, second)


def test_paired_statistics_align_reordered_population_and_singleton() -> None:
    baseline = (prediction(0), prediction(1))
    candidate = tuple(replace(row, probability=0.7) for row in reversed(baseline))
    result = paired_comparison(candidate, baseline)
    assert result.log_loss is not None
    assert result.log_loss.mean_difference == pytest.approx(-log(0.7) + log(0.5))
    assert paired_comparison(candidate[:1], baseline[1:]).log_loss is None


def test_holm_keeps_unsupported_tests_in_the_registered_family() -> None:
    from experiments.common.statistics import holm_adjust

    assert holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03}) == {"a": 0.03, "c": 0.06, "b": 0.06}
    assert holm_adjust({"a": 0.01, "b": None}) == {"a": 0.02, "b": None}
    with pytest.raises(ValueError):
        holm_adjust({"a": float("nan")})
