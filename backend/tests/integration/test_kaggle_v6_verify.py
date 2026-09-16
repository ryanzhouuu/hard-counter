import importlib.util
from pathlib import Path
from typing import Any

import pytest

from clash_sos.application.dataset_staging import StagingConfig
from clash_sos.application.dataset_verify import (
    KaggleV6DatasetVerifyError,
    verify_kaggle_v6_dataset,
)

CONFIG = StagingConfig(threads=1, memory_limit="256MB", batch_rows=8)


def _prepare_helpers() -> Any:
    path = Path(__file__).with_name("test_kaggle_v6_prepare.py")
    spec = importlib.util.spec_from_file_location("kaggle_v6_prepare_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("prepare helpers are unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verify_kaggle_v6_dataset_detects_tamper_without_rewrite(tmp_path: Path) -> None:
    helpers = _prepare_helpers()
    archive, parquet, mapping = helpers._create_prepare_archive(tmp_path)
    raw_manifest = tmp_path / "raw-audit.json"
    helpers._write_raw_manifest(archive, parquet, mapping, raw_manifest)
    destination = tmp_path / "dest"
    published = helpers._prepare(tmp_path, destination, archive, raw_manifest)
    manifest_before = published.joinpath("manifest.json").read_bytes()
    verify_kaggle_v6_dataset(published, config=CONFIG, temp_directory=tmp_path / "verify-tmp")
    canonical = published / "canonical.parquet"
    payload = bytearray(canonical.read_bytes())
    payload[-1] ^= 1
    canonical.write_bytes(payload)
    with pytest.raises(KaggleV6DatasetVerifyError, match="hash mismatch"):
        verify_kaggle_v6_dataset(
            published, config=CONFIG, temp_directory=tmp_path / "verify-tmp-tamper"
        )
    assert published.joinpath("manifest.json").read_bytes() == manifest_before
