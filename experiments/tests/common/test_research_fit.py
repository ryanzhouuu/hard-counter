from dataclasses import replace

import numpy as np
import torch
from experiments.common.contracts import OptimizerConfig
from experiments.common.data_access import ResearchRow, RoleAccess
from experiments.common.fit import fit_research
from experiments.matchup_features.model import ResearchModel
from experiments.tests.common.research_fixture import protocol, rows


def features(training: tuple[ResearchRow, ...], population: tuple[ResearchRow, ...]) -> np.ndarray:
    return np.array([[float(r.tokens[0][-1] - r.tokens[1][-1])] for r in population])


def factory(training: tuple[ResearchRow, ...], count: int) -> ResearchModel:
    return ResearchModel(17, count)


def test_fit_refit_deterministic_and_later_labels_cannot_change_outputs() -> None:
    population = rows()
    config = OptimizerConfig(max_epochs=2, patience=1, time_limit_seconds=20)
    source = RoleAccess(protocol(population), population)
    first = fit_research(source, features, factory, config, seed=0, penalty=0.001)
    changed = tuple(
        replace(r, label=1 - r.label) if i >= 12 else r for i, r in enumerate(population)
    )
    second = fit_research(
        RoleAccess(protocol(population), changed), features, factory, config, seed=0, penalty=0.001
    )
    assert first.selected_epoch == second.selected_epoch
    assert np.array_equal(first.scales, second.scales)
    for a, b in zip(first.model.parameters(), second.model.parameters(), strict=True):
        assert torch.equal(a, b)
    assert first.rows_per_second > 0 and first.peak_memory_bytes > 0
    assert np.array_equal(
        first.scales, np.sqrt(np.mean(features(population, population[:12]) ** 2, axis=0))
    )
