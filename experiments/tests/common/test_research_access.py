from dataclasses import replace
from pathlib import Path

import duckdb
import pytest
from experiments.common.data_access import RoleAccess, validate_rows
from experiments.common.protocol import search_protocol
from experiments.tests.common.research_fixture import protocol, rows


def test_role_reads_fail_before_returning_labels() -> None:
    population = rows()
    access = RoleAccess(protocol(population), population)
    assert len(access.read("refit", "fit")) == 12
    for role in ("calibration", "development", "reporting"):
        with pytest.raises(PermissionError):
            access.read(role, "fit")
    with pytest.raises(PermissionError):
        access.read("development", "calibrate")
    with pytest.raises(ValueError, match="exclude reporting"):
        RoleAccess(
            protocol(population).model_copy(update={"reporting": protocol(population).development}),
            population,
        )


def test_rows_reject_duplicate_reordered_and_missing_events() -> None:
    population = rows()
    for broken in (population[::-1], (*population, population[-1])):
        with pytest.raises(ValueError):
            validate_rows(broken)
    broken = list(population)
    broken[2] = replace(broken[2], event_key=broken[0].event_key)
    with pytest.raises(ValueError):
        validate_rows(broken)
    with pytest.raises(ValueError, match="digest mismatch"):
        RoleAccess(protocol(population), population[:-1])


def test_validation_split_keeps_timestamp_ties(tmp_path: Path) -> None:
    population = rows()
    original = protocol(population)
    with duckdb.connect() as db:
        db.execute(
            "CREATE TABLE canonical(timestamp TIMESTAMPTZ, fingerprint VARCHAR, "
            "archive_member VARCHAR, row_number BIGINT)"
        )
        db.executemany("INSERT INTO canonical VALUES(?,?,?,?)", [r.key for r in population])
        db.execute(
            "CREATE TABLE splits AS SELECT *, CASE WHEN row_number < 12 THEN 'train' "
            "ELSE 'validation' END AS partition FROM canonical"
        )
        db.execute("COPY canonical TO ? (FORMAT PARQUET)", [str(tmp_path / "canonical.parquet")])
        db.execute("COPY splits TO ? (FORMAT PARQUET)", [str(tmp_path / original.split_file)])
    boundary = population[18].key[0]
    resolved = search_protocol(
        original.model_copy(
            update={
                "development": original.development.model_copy(
                    update={"start": population[12].key[0]}
                )
            }
        ),
        tmp_path,
        calibration_end=boundary,
    )
    assert resolved.calibration is not None
    assert resolved.calibration.row_count == 6
    assert resolved.development.row_count == 6
    assert resolved.reporting is None
