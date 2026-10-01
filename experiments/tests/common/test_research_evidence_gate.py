from pathlib import Path

import pytest
from experiments.common import run, session
from experiments.mechanics.load import to_payload
from experiments.tests.common.comparison_fixture import comparison_fixture

from clash_sos.domain.canonical_dataset import canonical_json_bytes


def test_frozen_fit_rejects_synthetic_mechanics_before_fitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _, _, inputs = comparison_fixture(tmp_path)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("synthetic mechanics reached controlled fitting")

    monkeypatch.setattr(run, "fit_research", forbidden)
    result = run.run_variant(
        Path.cwd(),
        tmp_path / "rejected",
        config,
        config.variants[1],
        inputs.access,
        inputs.population,
        inputs.catalog,
        inputs.schema,
        seed=0,
        penalty=0.1,
        phase="screen",
    )
    assert result.status == "failed" and not result.eligible_for_comparison
    assert "synthetic mechanics" in result.failures[0]


def test_real_snapshot_rejects_operator_supplied_synthetic_mechanics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _, _, inputs = comparison_fixture(tmp_path)

    def loaded(*args: object, **kwargs: object) -> tuple[object, ...]:
        return inputs.access, inputs.population, inputs.schema

    monkeypatch.setattr(session, "load_snapshot", loaded)
    mechanics = tmp_path / "synthetic-mechanics.json"
    mechanics.write_bytes(canonical_json_bytes(to_payload(inputs.catalog)))
    with pytest.raises(ValueError, match="synthetic mechanics"):
        session.prepare_session(
            config,
            tmp_path / "inputs",
            synthetic=False,
            dataset=tmp_path / "snapshot",
            protocol=tmp_path / "protocol.json",
            schema=tmp_path / "schema.json",
            cache=tmp_path / "cache",
            mechanics=mechanics,
            row_cap=128,
        )
