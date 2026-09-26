"""Train and publish an attention artifact from resolved tiny-data slices."""

from collections.abc import Callable
from pathlib import Path

import pytest
from attention_cache_fixture import write_cache_dataset
from catalog_fixture import expanded_catalog

from clash_sos.application.attention_fit import AttentionFitConfig, AttentionFitTimeLimit
from clash_sos.application.dataset_staging import StagingConfig
from clash_sos.application.model_train_attention import (
    AttentionTrainError,
    train_attention_artifact,
)
from clash_sos.domain.attention_artifact import AttentionArtifactManifest
from clash_sos.domain.attention_protocol import AttentionProtocol
from clash_sos.domain.attention_schema import AttentionModelConfig, build_attention_schema
from clash_sos.domain.canonical_dataset import canonical_json_bytes
from clash_sos.infrastructure.ml.attention_artifact_io import load_attention_artifact


def inputs(tmp_path: Path) -> tuple[Path, Path, AttentionProtocol]:
    """Create real joined Parquet and a frozen protocol JSON beside it."""
    dataset = tmp_path / "dataset"
    _, protocol = write_cache_dataset(dataset)
    path = tmp_path / "protocol.json"
    path.write_bytes(canonical_json_bytes(protocol.model_dump(mode="python")))
    return dataset, path, protocol


def train(
    tmp_path: Path,
    dataset: Path,
    protocol_path: Path,
    *,
    network_config_path: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Apply small CPU settings to the public application runner."""
    return train_attention_artifact(
        dataset,
        tmp_path / "artifact",
        protocol_path=protocol_path,
        cache_directory=tmp_path / "cache",
        output_workspace=tmp_path / "output",
        fit_config=AttentionFitConfig(batch_size=1, max_epochs=1, seed=4, device="cpu"),
        staging_config=StagingConfig(chunk_size=1024, batch_rows=1, memory_limit="256MB"),
        model_version="attention-tiny-v1",
        network_config_path=network_config_path,
        progress=progress,
    )


def test_attention_training_publishes_evaluation_and_cleans_working_files(
    tmp_path: Path,
) -> None:
    dataset, protocol_path, protocol = inputs(tmp_path)
    progress: list[str] = []
    destination = train(tmp_path, dataset, protocol_path, progress=progress.append)
    manifest, _, model = load_attention_artifact(destination, chunk_size=1024)
    assert isinstance(manifest, AttentionArtifactManifest)
    assert manifest.fit_protocol == protocol
    assert model.training is False
    assert (destination / "evaluations/development/predictions-00000.parquet").is_file()
    assert (destination / "evaluations/reporting/predictions-00000.parquet").is_file()
    assert not (destination / "_working").exists()
    assert not (tmp_path / "output").exists()
    assert (tmp_path / "cache/manifest.json").is_file()
    assert progress[-1] == "publishing verified attention artifact"


def test_attention_training_rejects_existing_destination_before_fit(tmp_path: Path) -> None:
    dataset, protocol_path, _ = inputs(tmp_path)
    destination = tmp_path / "artifact"
    destination.mkdir()
    marker = destination / "keep"
    marker.write_text("unchanged")
    with pytest.raises(ValueError, match="already exists"):
        train(tmp_path, dataset, protocol_path)
    assert marker.read_text() == "unchanged"
    assert not (tmp_path / "cache").exists()


def test_attention_training_rejects_bad_protocol_and_network(tmp_path: Path) -> None:
    dataset, protocol_path, protocol = inputs(tmp_path)
    protocol_path.write_text("{}")
    with pytest.raises(AttentionTrainError, match="protocol is invalid"):
        train(tmp_path, dataset, protocol_path)
    assert not (tmp_path / "cache").exists()
    protocol_path.write_bytes(canonical_json_bytes(protocol.model_dump(mode="python")))
    network = tmp_path / "network.json"
    network.write_text('{"embedding_width":128}')
    with pytest.raises(AttentionTrainError, match="encoding does not match"):
        train(tmp_path, dataset, protocol_path, network_config_path=network)


def test_attention_training_cleans_output_after_fit_failure(tmp_path: Path) -> None:
    dataset, protocol_path, _ = inputs(tmp_path)
    with pytest.raises(AttentionFitTimeLimit):
        train_attention_artifact(
            dataset,
            tmp_path / "artifact",
            protocol_path=protocol_path,
            cache_directory=tmp_path / "cache",
            output_workspace=tmp_path / "output",
            fit_config=AttentionFitConfig(
                batch_size=1, max_epochs=1, seed=4, device="cpu", time_limit_seconds=1e-12
            ),
            staging_config=StagingConfig(chunk_size=1024, batch_rows=1, memory_limit="256MB"),
            model_version="attention-tiny-v1",
        )
    assert not (tmp_path / "artifact").exists()
    assert not (tmp_path / "output").exists()


def test_attention_training_freezes_explicit_card_inputs(tmp_path: Path) -> None:
    dataset, protocol_path, protocol = inputs(tmp_path)
    catalog, attributes = expanded_catalog()
    catalog_path, attributes_path = tmp_path / "catalog.json", tmp_path / "attributes.json"
    catalog_path.write_bytes(catalog.serialize())
    attributes_path.write_bytes(canonical_json_bytes(attributes.to_payload()))
    schema = build_attention_schema(
        catalog.serialize(), attributes=attributes, network=AttentionModelConfig()
    )
    protocol = protocol.model_copy(update={"encoding_sha256": schema.fingerprint()})
    protocol_path.write_bytes(canonical_json_bytes(protocol.model_dump(mode="python")))
    destination = train_attention_artifact(
        dataset,
        tmp_path / "artifact",
        protocol_path=protocol_path,
        cache_directory=tmp_path / "cache",
        output_workspace=tmp_path / "output",
        fit_config=AttentionFitConfig(batch_size=1, max_epochs=1, seed=4, device="cpu"),
        staging_config=StagingConfig(chunk_size=1024, batch_rows=1, memory_limit="256MB"),
        model_version="expanded-v1",
        catalog_path=catalog_path,
        attributes_path=attributes_path,
    )
    manifest, frozen, model = load_attention_artifact(destination, chunk_size=1024)
    assert manifest.catalog_version == catalog.version
    assert frozen == schema
    assert model.identity_count == 266
    assert frozen.attribute_snapshot == attributes.to_payload()
