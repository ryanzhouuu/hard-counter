import importlib.util
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import pytest

from clash_sos.application.dataset_staging import StagingConfig
from clash_sos.application.dataset_verify import (
    KaggleV6DatasetVerifyError,
    verify_kaggle_v6_dataset,
)
from clash_sos.domain.processed_manifest import (
    ProcessedDatasetManifest,
    ProcessedOutputFile,
    dump_processed_manifest,
)
from clash_sos.infrastructure.kaggle_v6.audit_io import hash_file

CONFIG = StagingConfig(threads=1, memory_limit="256MB", chunk_size=1024)
LOGICAL_SHA256 = "b" * 64
FILE_KINDS: dict[
    str,
    Literal[
        "canonical",
        "disposition",
        "player_disjoint_split",
        "temporal_split",
        "verification_report",
    ],
] = {
    "canonical.parquet": "canonical",
    "dispositions/member-a.parquet": "disposition",
    "splits-player-disjoint.parquet": "player_disjoint_split",
    "splits-temporal.parquet": "temporal_split",
    "verification-report.json": "verification_report",
}


def _manifest_helpers() -> Any:
    path = Path(__file__).with_name("test_processed_manifest.py")
    spec = importlib.util.spec_from_file_location("processed_manifest_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("processed manifest helpers are unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _write_published_tree(root: Path) -> ProcessedDatasetManifest:
    helpers = _manifest_helpers()
    files: list[ProcessedOutputFile] = []
    for relative, payload in (
        ("canonical.parquet", b"canonical"),
        ("dispositions/member-a.parquet", b"disposition"),
        ("splits-player-disjoint.parquet", b"player"),
        ("splits-temporal.parquet", b"temporal"),
        ("verification-report.json", b"report"),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        size, digest = hash_file(path, CONFIG.chunk_size)
        kind = FILE_KINDS[relative]
        files.append(
            ProcessedOutputFile(
                path=relative,
                kind=kind,
                size_bytes=size,
                sha256=digest,
                logical_sha256=LOGICAL_SHA256 if kind == "canonical" else None,
            )
        )
    base = helpers.processed_manifest()
    assert isinstance(base, ProcessedDatasetManifest)
    manifest_payload = base.model_dump(mode="python")
    manifest_payload["files"] = [file.model_dump(mode="python") for file in files]
    manifest = ProcessedDatasetManifest.model_validate(manifest_payload)
    (root / "manifest.json").write_bytes(dump_processed_manifest(manifest))
    return manifest


def test_verify_kaggle_v6_dataset_requires_manifest(tmp_path: Path) -> None:
    with pytest.raises(KaggleV6DatasetVerifyError, match="processed manifest is required"):
        verify_kaggle_v6_dataset(tmp_path, config=CONFIG, temp_directory=tmp_path / "tmp")


def test_verify_kaggle_v6_dataset_accepts_matching_hashes(tmp_path: Path) -> None:
    version = tmp_path / "dataset"
    expected = _write_published_tree(version)
    before = _tree_bytes(version)
    with patch("clash_sos.application.dataset_verify.verify_processed_artifacts") as duckdb_verify:
        manifest = verify_kaggle_v6_dataset(version, config=CONFIG, temp_directory=tmp_path / "tmp")
    assert manifest == expected
    duckdb_verify.assert_called_once()
    assert _tree_bytes(version) == before


def test_verify_kaggle_v6_dataset_rejects_tampered_file_before_duckdb(tmp_path: Path) -> None:
    version = tmp_path / "dataset"
    _write_published_tree(version)
    canonical = version / "canonical.parquet"
    canonical.write_bytes(canonical.read_bytes() + b"x")
    before = _tree_bytes(version)
    with (
        patch("clash_sos.application.dataset_verify.verify_processed_artifacts") as duckdb_verify,
        pytest.raises(KaggleV6DatasetVerifyError, match="hash mismatch"),
    ):
        verify_kaggle_v6_dataset(version, config=CONFIG, temp_directory=tmp_path / "tmp")
    duckdb_verify.assert_not_called()
    assert _tree_bytes(version) == before
