"""Validate the separate attention artifact inventory contract."""

from pathlib import Path

import pytest
from attention_cache_fixture import write_cache_dataset
from pydantic import ValidationError

from clash_sos.domain.attention_artifact import (
    AttentionArtifactFile,
    AttentionArtifactManifest,
    AttentionArtifactRuntime,
    dump_attention_manifest,
)


def manifest(tmp_path: Path) -> AttentionArtifactManifest:
    """Build an artifact identity from a resolved tiny fit population."""
    schema, protocol = write_cache_dataset(tmp_path / "dataset")
    files = (
        AttentionArtifactFile(
            path="card-catalog.json", kind="card_catalog", size_bytes=1, sha256="a" * 64
        ),
        AttentionArtifactFile(
            path="evaluation.json", kind="evaluation", size_bytes=1, sha256="a" * 64
        ),
        AttentionArtifactFile(
            path="feature-schema.json", kind="feature_schema", size_bytes=1, sha256="a" * 64
        ),
        AttentionArtifactFile(path="weights.pt", kind="weights", size_bytes=1, sha256="a" * 64),
    )
    return AttentionArtifactManifest(
        model_version="attention-test-v1",
        dataset_version=protocol.dataset_version,
        catalog_version=schema.catalog_version,
        balance_era_id=protocol.balance_era_id,
        encoding_sha256=schema.fingerprint(),
        fit_protocol=protocol,
        fit_seed=7,
        runtime=AttentionArtifactRuntime(device="cpu", python_version="3.12", torch_version="2.13"),
        files=files,
    )


def test_attention_manifest_round_trips_with_distinct_discriminator(tmp_path: Path) -> None:
    published = manifest(tmp_path)
    restored = AttentionArtifactManifest.model_validate_json(dump_attention_manifest(published))
    assert restored == published
    assert restored.manifest_type == "matchup_attention"


def test_attention_manifest_rejects_missing_file_and_mixed_protocol(tmp_path: Path) -> None:
    published = manifest(tmp_path)
    with pytest.raises(ValidationError, match="required file kind"):
        AttentionArtifactManifest.model_validate(
            {**published.model_dump(mode="python"), "files": published.files[:-1]}
        )
    with pytest.raises(ValidationError, match="identities disagree"):
        AttentionArtifactManifest.model_validate(
            {**published.model_dump(mode="python"), "balance_era_id": "different"}
        )


def test_attention_manifest_rejects_uninventoried_member_kind(tmp_path: Path) -> None:
    published = manifest(tmp_path)
    extra = AttentionArtifactFile(
        path="extra.json", kind="predictions", size_bytes=1, sha256="b" * 64
    )
    with pytest.raises(ValidationError, match="unexpected member"):
        AttentionArtifactManifest.model_validate(
            {
                **published.model_dump(mode="python"),
                "files": tuple(sorted((*published.files, extra), key=lambda item: item.path)),
            }
        )
