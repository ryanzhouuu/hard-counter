import importlib.util
from pathlib import Path
from shutil import copytree

import duckdb

from clash_sos.application.dataset_grouping import group_staged_dataset
from clash_sos.application.dataset_materialize import materialize_canonical_dataset
from clash_sos.application.dataset_staging import StagingConfig, stage_kaggle_v6
from clash_sos.domain.canonical_dataset import CANONICAL_SCHEMA
from clash_sos.infrastructure.kaggle_v6.materialize_io import read_canonical_schema


def _create_grouping_archive(directory: Path) -> Path:
    path = Path(__file__).with_name("test_kaggle_v6_grouping.py")
    spec = importlib.util.spec_from_file_location("kaggle_v6_grouping_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("grouping archive helper is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_grouping_archive(directory)


def _stage_and_group(tmp_path: Path) -> tuple[Path, StagingConfig]:
    workspace = tmp_path / "workspace"
    config = StagingConfig(batch_rows=2, threads=1, memory_limit="256MB")
    staged = stage_kaggle_v6(
        _create_grouping_archive(tmp_path),
        workspace,
        temp_directory=tmp_path / "stage-tmp",
        config=config,
    )
    group_staged_dataset(
        workspace,
        workspace,
        source_row_count=staged.source_row_count,
        config=config,
        temp_directory=tmp_path / "group-tmp",
    )
    return workspace, config


def test_materialize_synthetic_archive_twice(tmp_path: Path) -> None:
    workspace, config = _stage_and_group(tmp_path)
    first = tmp_path / "workspace-a"
    second = tmp_path / "workspace-b"
    copytree(workspace / "dispositions", first / "dispositions")
    copytree(workspace / "dispositions", second / "dispositions")
    result_a = materialize_canonical_dataset(
        workspace, first, config=config, temp_directory=tmp_path / "tmp-a"
    )
    result_b = materialize_canonical_dataset(
        workspace, second, config=config, temp_directory=tmp_path / "tmp-b"
    )
    assert result_a.row_count == 1
    assert result_a.logical_sha256 == result_b.logical_sha256
    assert result_a.sha256 == result_b.sha256
    assert result_a.canonical_path.read_bytes() == result_b.canonical_path.read_bytes()
    connection = duckdb.connect()
    try:
        schema = read_canonical_schema(connection, result_a.canonical_path)
        observed = tuple((column.name, column.physical_type) for column in schema)
        expected = tuple((column.name, column.physical_type) for column in CANONICAL_SCHEMA)
        assert observed == expected
        rows = connection.execute(
            "SELECT archive_member, row_number FROM read_parquet(?)",
            [str(result_a.canonical_path)],
        ).fetchall()
        excluded = connection.execute(
            """
            SELECT COUNT(*) FROM read_parquet(?)
            WHERE (archive_member, row_number) IN (
                ('a.parquet', 1), ('a.parquet', 2), ('a.parquet', 6),
                ('b.parquet', 3), ('b.parquet', 4), ('b.parquet', 5)
            )
            """,
            [str(result_a.canonical_path)],
        ).fetchone()
    finally:
        connection.close()
    assert rows == [("a.parquet", 0)]
    assert excluded == (0,)
