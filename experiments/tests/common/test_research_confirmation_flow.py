import argparse
import json
import sqlite3
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest
from experiments.common import reporting_consumption
from experiments.common.artifacts import file_record, load_run
from experiments.common.candidate import freeze_candidate
from experiments.common.comparison import comparison_report
from experiments.common.confirm import confirm
from experiments.common.contracts import StudyConfig, fingerprint
from experiments.common.data_access import ReportingContract
from experiments.common.provenance import code_digest
from experiments.common.source_rows import encode_official_row, population_identity
from experiments.tests.common.comparison_fixture import comparison_fixture
from pydantic_core import to_jsonable_python
from test_canonical_dataset import valid_row_payload

from clash_sos.domain.attention_dataset import TowerBattleRowV2
from clash_sos.domain.canonical_dataset import canonical_json_bytes, deck_content_hash


@pytest.mark.parametrize("corrupt_source", [False, True])
def test_frozen_ensembles_confirm_once_without_fit_or_calibration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corrupt_source: bool,
) -> None:
    def consumption_path(_root: Path) -> Path:
        return tmp_path / "consumption.db"

    monkeypatch.setattr(reporting_consumption, "registry_path", consumption_path)
    config, models, reports, session = comparison_fixture(tmp_path)
    comparison_report(config, {"A1": 0.1}, models, reports)
    frozen_at = datetime(2026, 9, 2, tzinfo=UTC)
    reporting_start = datetime(2026, 9, 3, tzinfo=UTC)
    original = session.access.read("refit", "fit")[0]
    schema = session.schema
    fields: dict[str, object] = {}
    for side, tokens in zip(("a", "b"), original.tokens, strict=True):
        identities = tuple(schema.identity_vocab[token].rsplit(":", 1) for token in tokens[:8])
        cards, forms = tuple(item[0] for item in identities), tuple(item[1] for item in identities)
        levels = (16,) * 8
        fields.update(
            {
                f"side_{side}_card_ids": cards,
                f"side_{side}_card_forms": forms,
                f"side_{side}_card_levels": levels,
                f"side_{side}_deck_hash": deck_content_hash(cards, forms, levels),
                f"side_{side}_tower": schema.identity_vocab[tokens[-1]],
                f"side_{side}_tower_level": 16,
            }
        )
    row = TowerBattleRowV2.model_validate(
        valid_row_payload(
            **fields,
            source_id="official-api",
            dataset_version="future-synthetic",
            mode="Ranked1v1_NewArena2",
            balance_era_id=schema.balance_era_id,
            timestamp=reporting_start,
            event_key=sha256(b"future-event").hexdigest(),
            fingerprint=sha256(b"future-battle").hexdigest(),
            side_a_player_id=original.player_a,
            side_b_player_id=original.player_b,
            archive_member="future.jsonl",
            row_number=0,
        )
    )
    source = tmp_path / "future.jsonl"
    source.write_text(row.model_dump_json() + "\n")
    future = population_identity(
        (encode_official_row(row, schema, 0),),
        (file_record(source, tmp_path, row_count=1),),
        schema,
        0,
        start=reporting_start,
        end=datetime(2026, 9, 4, tzinfo=UTC),
    )
    frozen = freeze_candidate(
        config,
        {"A1": 0.1},
        code_sha256=code_digest(Path.cwd()),
        assets=tuple(
            file_record(reports / name, reports)
            for name in ("ensemble-A0.json", "ensemble-A1.json")
        ),
        frozen_at=frozen_at,
        inspected_through=session.population.end,
        reporting_start=reporting_start,
        confirmation_rules=config.rules.model_copy(
            update={"confirmation": "single", "test_alternative": "two_sided"}
        ),
    )
    freeze_path = reports / "freeze.json"
    freeze_path.write_bytes(canonical_json_bytes(frozen.model_dump(mode="json")))
    contract = ReportingContract(
        future, session.population, fingerprint(frozen), session.population.end, frozen_at
    )
    contract_path = reports / "reporting-contract.json"
    contract_path.write_bytes(canonical_json_bytes(to_jsonable_python(contract)))
    reporting = StudyConfig.model_validate(
        {
            **config.model_dump(),
            "stage": "prospective-reporting",
            "prospective": future,
            "rules": frozen.confirmation_rules,
        }
    )
    args = argparse.Namespace(
        freeze=freeze_path,
        reporting_contract=contract_path,
        reporting_source=source,
        output=tmp_path / "confirmation",
        run_id="first",
        row_cap=1,
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("prospective confirmation cannot fit or calibrate")

    monkeypatch.setattr("experiments.common.calibration.fit_temperature", forbidden)
    monkeypatch.setattr("experiments.common.fit.fit_research", forbidden)
    from experiments.tests.common.research_fixture import HASH

    def fixture_revision(_root: Path) -> tuple[str, None, str]:
        return "b" * 40, None, HASH

    monkeypatch.setattr("experiments.common.confirm.revision", fixture_revision)
    destination = args.output / config.study_id / args.run_id
    for change in ({"test_alternative": "improvement"}, {"confirmation": "holm"}):
        changed = StudyConfig.model_validate(
            {**reporting.model_dump(), "rules": {**reporting.rules.model_dump(), **change}}
        )
        with pytest.raises(ValueError, match="frozen confirmation rules"):
            confirm(args, changed)
        assert not destination.exists()
    if corrupt_source:
        source.write_text("corrupt")
        with pytest.raises(ValueError):
            confirm(args, reporting)
        failed = load_run(destination)
        assert failed.status == "failed" and failed.failures
        assert not failed.eligible_for_comparison
        with sqlite3.connect(tmp_path / "consumption.db") as connection:
            assert connection.execute("SELECT status FROM populations").fetchone() == ("failed",)
        args.run_id = "retry"
        with pytest.raises(FileExistsError):
            confirm(args, reporting)
        copied = freeze_path.with_name("copied-freeze.json")
        copied.write_bytes(freeze_path.read_bytes())
        args.freeze = copied
        args.output = tmp_path / "another-output"
        with pytest.raises(FileExistsError, match="consumed"):
            confirm(args, reporting)
        return
    confirm(args, reporting)
    manifest = load_run(destination)
    assert manifest.status == "complete" and not manifest.eligible_for_comparison
    assert manifest.config_sha256 == fingerprint(reporting)
    assert "candidate/candidate-freeze.json" in {entry.path for entry in manifest.inputs}
    result_path = args.output / config.study_id / args.run_id / "confirmation.json"
    result = json.loads(result_path.read_bytes())
    assert result["metrics"]["A0"]["metrics"]["row_count"] == 1
    assert "A1" in result["comparisons"]
    assert result["comparison_family"] == ["A1"]
    assert result["test_alternative"] == "two_sided"
    with sqlite3.connect(tmp_path / "consumption.db") as connection:
        assert connection.execute("SELECT status FROM populations").fetchone() == ("complete",)
    args.run_id = "second"
    with pytest.raises(FileExistsError):
        confirm(args, reporting)
    copied = freeze_path.with_name("copied-freeze.json")
    copied.write_bytes(freeze_path.read_bytes())
    args.freeze = copied
    args.output = tmp_path / "another-output"
    with pytest.raises(FileExistsError, match="consumed"):
        confirm(args, reporting)
