"""Development diagnostics describe nuisance effects without changing fitted assets."""

from experiments.common.data_access import RoleAccess
from experiments.common.predictions import Prediction, pair_predictions
from experiments.common.statistics import score_predictions
from experiments.matchup_features.model import ResearchModel
from experiments.player_adjustment.diagnostics import (
    evaluation_support,
    matchup_probability_stability,
    residual_player_diagnostic,
    shrinkage_diagnostics,
    support_bin_counts,
    training_diagnostics,
)
from experiments.player_adjustment.history import training_history
from experiments.player_adjustment.model import PlayerModel
from experiments.player_adjustment.support_contracts import SupportBins


def development_diagnostics(
    access: RoleAccess,
    model: ResearchModel,
    matchup: tuple[Prediction, ...],
    actual: tuple[Prediction, ...],
    bins: SupportBins | None,
    penalty: float,
) -> dict[str, object]:
    """Use refit covariates for support and development outcomes only for residuals."""
    refit = access.read("refit", "fit")
    development = access.read("development", "compare")
    indexed = {a.row_key: (a, b) for a, b in pair_predictions(matchup, actual)}
    if set(indexed) != {row.key for row in development}:
        raise ValueError("player diagnostics require the complete development population")
    match_p = tuple(indexed[row.key][0].probability for row in development)
    actual_p = tuple(indexed[row.key][1].probability for row in development)
    _, history = training_history(refit)
    training = training_diagnostics(refit)
    support = evaluation_support(development, history, training)
    nuisance: dict[str, object] = {"branch": "none", "penalty": penalty, "penalty_value": 0.0}
    if isinstance(model, PlayerModel):
        nuisance.update(
            branch=model.branch,
            penalty_value=float(model.nuisance_penalty(penalty).detach().cpu().item()),
        )
        if model.player_effects is not None:
            nuisance.update(shrinkage_diagnostics(model.player_effects.freeze()))
        else:
            nuisance.update(beta=float(model.beta.detach().cpu().item()), penalty_normalization=2.0)
    return {
        "interpretation": "descriptive development diagnostics; equal skill is not identified",
        "training": training,
        "evaluation_support": tuple(
            {"row_key": row.key, "event_key": row.event_key, "support": item}
            for row, item in zip(development, support, strict=True)
        ),
        "support_bins": bins,
        "support_bin_counts": support_bin_counts(development, history, training, bins)
        if bins is not None
        else None,
        "nuisance_shrinkage": nuisance,
        "actual_outcome_metrics": score_predictions(actual),
        "matchup_vs_actual_stability": matchup_probability_stability(match_p, actual_p),
        "matchup_residuals": residual_player_diagnostic(development, match_p, role="development"),
        "actual_outcome_residuals": residual_player_diagnostic(
            development, actual_p, role="development"
        ),
    }
