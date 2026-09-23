"""Exercise checked attention publication and CPU reload on real tiny cache rows."""

from json import loads
from pathlib import Path

import pytest
import torch
from test_attention_cache_read import write_cache

from clash_sos.application.attention_evaluate import (
    AttentionEvaluationReport,
    evaluate_attention_cache,
)
from clash_sos.application.attention_fit import AttentionFitConfig, fit_attention_model
from clash_sos.application.attention_support import AttentionSupportIndex
from clash_sos.domain.attention_artifact import AttentionArtifactManifest, dump_attention_manifest
from clash_sos.domain.canonical_dataset import canonical_json_bytes
from clash_sos.infrastructure.kaggle_v6.attention_cache_read import load_attention_cache
from clash_sos.infrastructure.kaggle_v6.audit_io import hash_file
from clash_sos.infrastructure.ml.attention_artifact_io import (
    AttentionArtifactError,
    finalize_attention_artifact,
    load_attention_artifact,
)


def publish_fixture(tmp_path: Path) -> Path:
    """Fit and score five joined rows through the production cache boundary."""
    dataset, directory, schema, protocol = write_cache(tmp_path)
    cache = load_attention_cache(dataset, directory, protocol, schema, chunk_size=1024)
    fit = fit_attention_model(
        schema, cache, AttentionFitConfig(batch_size=1, max_epochs=1, seed=3, device="cpu")
    )
    workspace = tmp_path / "output"
    workspace.mkdir()
    support = AttentionSupportIndex.build(cache, tmp_path / "support.db")
    try:
        reports: list[AttentionEvaluationReport] = []
        for role in ("development", "reporting"):
            evaluation = workspace / "evaluations" / role
            report = evaluate_attention_cache(
                cache,
                fit.refit_model,
                fit_protocol=protocol,
                fit_artifact_id="attention-test-v1",
                role=role,
                support=support,
                output_directory=evaluation,
                batch_size=1,
            )
            (evaluation / "report.json").unlink()
            reports.append(report)
    finally:
        support.close()
    destination = tmp_path / "artifact"
    finalize_attention_artifact(
        workspace,
        destination,
        schema=schema,
        protocol=protocol,
        fit=fit,
        reports=reports,
        model_version="attention-test-v1",
        chunk_size=1024,
    )
    _, _, restored = load_attention_artifact(destination, chunk_size=1024)
    first = torch.tensor(next(cache.iter_batches("development", batch_size=1))[0], dtype=torch.long)
    with torch.inference_mode():
        torch.testing.assert_close(
            torch.sigmoid(restored(first)), torch.sigmoid(fit.refit_model(first)), rtol=0, atol=0
        )
    return destination


def rewrite_inventory(path: Path, member: str) -> None:
    """Keep a tampered member's checksum valid to test deeper reload checks."""
    manifest = AttentionArtifactManifest.model_validate_json((path / "manifest.json").read_bytes())
    size, digest = hash_file(path / member, 1024)
    files = tuple(
        item.model_copy(update={"size_bytes": size, "sha256": digest})
        if item.path == member
        else item
        for item in manifest.files
    )
    (path / "manifest.json").write_bytes(
        dump_attention_manifest(manifest.model_copy(update={"files": files}))
    )


def test_attention_artifact_round_trip_and_refuses_existing_destination(tmp_path: Path) -> None:
    destination = publish_fixture(tmp_path)
    manifest, schema, model = load_attention_artifact(destination, chunk_size=1024)
    assert manifest.manifest_type == "matchup_attention"
    assert manifest.fit_seed == 3
    assert manifest.encoding_sha256 == schema.fingerprint() == model.schema_fingerprint
    assert not model.training
    assert {item.kind for item in manifest.files} == {
        "card_catalog",
        "feature_schema",
        "evaluation",
        "weights",
        "predictions",
    }
    assert (destination / "evaluations/development/predictions-00000.parquet").is_file()
    assert not (tmp_path / "output").exists()


def test_attention_artifact_rejects_corrupt_and_missing_files(tmp_path: Path) -> None:
    destination = publish_fixture(tmp_path)
    with (destination / "weights.pt").open("ab") as output:
        output.write(b"corruption")
    with pytest.raises(AttentionArtifactError, match="inventory mismatch"):
        load_attention_artifact(destination, chunk_size=1024)
    (destination / "weights.pt").unlink()
    with pytest.raises(AttentionArtifactError, match="inventory mismatch"):
        load_attention_artifact(destination, chunk_size=1024)


def test_attention_artifact_rejects_nonfinite_and_wrong_weight_shape(tmp_path: Path) -> None:
    destination = publish_fixture(tmp_path)
    weights = torch.load(destination / "weights.pt", map_location="cpu", weights_only=True)
    name = next(key for key, value in weights.items() if value.is_floating_point())
    weights[name].flatten()[0] = float("nan")
    torch.save(weights, destination / "weights.pt")
    rewrite_inventory(destination, "weights.pt")
    with pytest.raises(AttentionArtifactError, match="tensor is invalid"):
        load_attention_artifact(destination, chunk_size=1024)
    weights[name] = weights[name].reshape(-1)[:1]
    torch.save(weights, destination / "weights.pt")
    rewrite_inventory(destination, "weights.pt")
    with pytest.raises(AttentionArtifactError, match="tensor is invalid"):
        load_attention_artifact(destination, chunk_size=1024)


def test_attention_artifact_rejects_incompatible_schema_and_fit_record(tmp_path: Path) -> None:
    destination = publish_fixture(tmp_path)
    schema_path = destination / "feature-schema.json"
    schema = loads(schema_path.read_bytes())
    schema["network"]["dropout"] = 0.1
    schema_path.write_bytes(canonical_json_bytes(schema) + b"\n")
    rewrite_inventory(destination, "feature-schema.json")
    with pytest.raises(AttentionArtifactError, match="schema or catalog mismatch"):
        load_attention_artifact(destination, chunk_size=1024)

    schema["network"]["dropout"] = 0.0
    schema_path.write_bytes(canonical_json_bytes(schema) + b"\n")
    rewrite_inventory(destination, "feature-schema.json")
    evaluation_path = destination / "evaluation.json"
    evaluation = loads(evaluation_path.read_bytes())
    evaluation["fit"]["config"]["seed"] = 99
    evaluation_path.write_bytes(canonical_json_bytes(evaluation) + b"\n")
    rewrite_inventory(destination, "evaluation.json")
    with pytest.raises(AttentionArtifactError, match="fit provenance is invalid"):
        load_attention_artifact(destination, chunk_size=1024)
