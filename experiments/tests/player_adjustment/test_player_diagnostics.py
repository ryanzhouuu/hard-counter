import pytest
from experiments.player_adjustment.diagnostics import (
    SupportBins,
    evaluation_support,
    matchup_probability_stability,
    residual_player_diagnostic,
    shrinkage_diagnostics,
    support_bin_counts,
    training_diagnostics,
)
from experiments.player_adjustment.history import training_history
from experiments.player_adjustment.joint import JointPlayerEffects
from experiments.tests.player_adjustment.test_player_history import row


def test_diagnostics_preserve_seen_missing_and_connectivity() -> None:
    rows = [row(0), row(1), row(2, a="c", b="d")]
    _, history = training_history(rows)
    diagnostics = training_diagnostics(rows)
    assert diagnostics.component_sizes == (2, 2)
    assert all(item.dominant_deck_share == 1 for item in diagnostics.players)
    assert any("disconnected" in warning for warning in diagnostics.warnings)
    evaluation = [row(3, a="a", b="c"), row(4, a="a", b="new")]
    support = evaluation_support(evaluation, history, diagnostics)
    assert support[0].same_training_component is False
    assert support[0].player_a_prior_count == 2
    assert support[1].same_training_component is None
    assert not support[1].player_b_seen
    assert support[1].player_b_missing_history
    frozen = JointPlayerEffects.from_training(rows).freeze()
    assert shrinkage_diagnostics(frozen)["mean_squared_effect"] == 0


def test_residual_skill_is_development_only_and_descriptive() -> None:
    diagnostics = residual_player_diagnostic([row(0)], [0.6], role="development")
    assert diagnostics[0].mean_residual == pytest.approx(0.4)
    assert diagnostics[1].mean_residual == pytest.approx(-0.4)
    assert all(item.interpretation == "descriptive-development-residual" for item in diagnostics)
    with pytest.raises(ValueError, match="development"):
        residual_player_diagnostic([row(0)], [0.6], role="reporting")
    with pytest.raises(ValueError, match="aligned"):
        residual_player_diagnostic([row(0)], [], role="development")


def test_frozen_support_bins_and_probability_stability() -> None:
    training = [row(0), row(1)]
    _, history = training_history(training)
    summary = training_diagnostics(training)
    bins = SupportBins(prior_history_edges=(1, 2, 5), deck_switching_edges=(1, 2))
    support = support_bin_counts([row(3, b="new")], history, summary, bins)
    assert support["prior_history"] == (1, 0, 1, 0)
    assert support["deck_switching"] == (2, 0, 0)
    assert support["seen_pair"] == (0, 0, 1, 0)
    assert SupportBins.model_validate_json(bins.model_dump_json()) == bins
    with pytest.raises(ValueError, match="boundaries"):
        SupportBins(prior_history_edges=(2, 1), deck_switching_edges=(1,))
    stability = matchup_probability_stability([0.2, 0.8], [0.3, 0.6])
    assert stability["mean_absolute_probability_change"] == pytest.approx(0.15)
    with pytest.raises(ValueError, match="aligned"):
        matchup_probability_stability([0.2], [])
