import pytest
from experiments.common.contracts import OptimizerConfig
from experiments.model_diagnostics.schedule import (
    baseline_jobs,
    branch_jobs,
    finalist_jobs,
    player_jobs,
    select_job,
)


def test_staged_grid_stays_within_budget_and_separates_penalties() -> None:
    jobs = baseline_jobs(OptimizerConfig(max_epochs=80, patience=8))
    assert len(jobs) == 13
    assert len({j.name for j in jobs}) == len(jobs)
    reference = jobs[0]
    assert reference.spec.optimizer.max_epochs == 20
    assert reference.spec.optimizer.patience == 3
    assert reference.spec.model.card_l2 == reference.spec.model.pair_l2 == 0
    backbone = jobs[-1]
    players = player_jobs(backbone)
    assert len(players) == 6
    for job in players:
        assert job.spec.model.card_l2 == backbone.spec.model.card_l2
        assert job.spec.model.pair_l2 == backbone.spec.model.pair_l2
        assert job.spec.optimizer == backbone.spec.optimizer
    history = branch_jobs(players, "history")
    joint = branch_jobs(players, "joint")
    assert len(history) == len(joint) == 3
    finalists = finalist_jobs(backbone, history[0], joint[0])
    assert len(finalists) == 6
    assert {j.spec.seed for j in finalists} == {1, 2}
    assert len(jobs) + len(players) + len(finalists) + 1 == 26


def test_selection_requires_complete_valid_grid_and_has_fixed_tie_break() -> None:
    jobs = baseline_jobs(OptimizerConfig())
    losses: dict[str, float | None] = {j.name: 0.69 for j in jobs}
    assert select_job(jobs, losses) == jobs[0]
    losses[jobs[0].name] = None
    assert select_job(jobs, losses) == jobs[1]
    with pytest.raises(ValueError, match="every registered"):
        select_job(jobs, {})
    with pytest.raises(ValueError, match="eligible"):
        select_job(jobs, {j.name: None for j in jobs})
    with pytest.raises(ValueError, match="finite"):
        select_job(jobs, {j.name: float("nan") for j in jobs})
