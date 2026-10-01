from pathlib import Path

import pytest
from experiments.common import selection
from experiments.common.matrix import screen_jobs
from experiments.tests.common.research_fixture import HASH
from experiments.tests.common.test_research_matrix import (
    DevelopmentAccessSpy,
    frozen_config,
    result_run,
)


def test_selection_resume_does_not_reopen_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def revision(_root: Path) -> tuple[str, None, str]:
        return "b" * 40, None, HASH

    def digest(_root: Path) -> str:
        return HASH

    def changed(_root: Path) -> str:
        return "d" * 64

    monkeypatch.setattr(selection, "revision", revision)
    monkeypatch.setattr(selection, "code_digest", digest)
    config = frozen_config()
    results = [
        result_run(tmp_path, config, job, confidence)
        for job, confidence in zip(screen_jobs(config)[1:], (0.8, 0.95, 0.85), strict=True)
    ]
    decision = tmp_path / "selection.json"
    first = selection.select_once(config, results, DevelopmentAccessSpy(), decision, tmp_path)
    untouched = DevelopmentAccessSpy()
    assert selection.select_once(config, results, untouched, decision, tmp_path) == first
    assert not untouched.reads
    monkeypatch.setattr(selection, "code_digest", changed)
    with pytest.raises(ValueError, match="current code"):
        selection.select_once(config, results, untouched, decision, tmp_path)
    assert not untouched.reads


def test_selection_refuses_dirty_checkout_before_reading_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def dirty_revision(_root: Path) -> tuple[str, str, str]:
        return "b" * 40, HASH, HASH

    monkeypatch.setattr(selection, "revision", dirty_revision)
    access = DevelopmentAccessSpy()
    with pytest.raises(ValueError, match="clean"):
        selection.select_once(frozen_config(), [], access, tmp_path / "choice.json", tmp_path)
    assert not access.reads
