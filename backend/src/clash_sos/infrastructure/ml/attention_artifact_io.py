"""Publish and reload checked attention weights without touching legacy artifacts.

Training owns the temporary workspace; this module inventories its completed
members and verifies the entire published artifact before returning a model.
"""

from collections.abc import Sequence
from dataclasses import asdict
from json import loads
from pathlib import Path
from pickle import UnpicklingError
from typing import cast

import torch
from pydantic import ValidationError
from torch import Tensor

from clash_sos.application.attention_evaluate import AttentionEvaluationReport
from clash_sos.application.attention_fit import AttentionFitConfig, AttentionFitResult
from clash_sos.domain.attention_artifact import (
    REQUIRED_ATTENTION_FILES,
    AttentionArtifactFile,
    AttentionArtifactManifest,
    AttentionArtifactRuntime,
    dump_attention_manifest,
)
from clash_sos.domain.attention_model import AttentionMatchupModel
from clash_sos.domain.attention_protocol import AttentionProtocol
from clash_sos.domain.attention_schema import AttentionCardSchema
from clash_sos.domain.canonical_dataset import canonical_json_bytes
from clash_sos.infrastructure.kaggle_v6.audit_io import hash_file
from clash_sos.infrastructure.kaggle_v6.publish_io import publish_processed_version


class AttentionArtifactError(ValueError):
    """An attention artifact is incomplete, corrupt, or incompatible."""


def _inventory(workspace: Path, chunk_size: int) -> tuple[AttentionArtifactFile, ...]:
    """Hash only allowed members and reject hidden or uninventoried output."""
    files: list[AttentionArtifactFile] = []
    for path in sorted(workspace.rglob("*")):
        if path.is_symlink():
            raise AttentionArtifactError(f"artifact member is unsafe: {path}")
        if path.is_dir():
            continue
        relative = path.relative_to(workspace).as_posix()
        if relative == "manifest.json":
            continue
        if not path.is_file():
            raise AttentionArtifactError(f"artifact member is unsafe: {relative}")
        if relative in REQUIRED_ATTENTION_FILES:
            kind = REQUIRED_ATTENTION_FILES[relative]
        elif relative.startswith("evaluations/") and relative.endswith(".parquet"):
            kind = "predictions"
        else:
            raise AttentionArtifactError(f"artifact member is not declared: {relative}")
        size, digest = hash_file(path, chunk_size)
        files.append(
            AttentionArtifactFile.model_validate(
                {"path": relative, "kind": kind, "size_bytes": size, "sha256": digest}
            )
        )
    return tuple(files)


def _evaluation_payload(
    fit: AttentionFitResult, reports: Sequence[AttentionEvaluationReport]
) -> dict[str, object]:
    """Keep selection history with the immutable scored-row reports."""
    return {
        "fit": {
            "config": fit.config.model_dump(mode="python"),
            "selected_epoch": fit.selected_epoch,
            "selection_reason": fit.selection_reason,
            "history": [asdict(item) for item in fit.history],
            "elapsed_seconds": fit.elapsed_seconds,
            "trained_rows": fit.trained_rows,
            "rows_per_second": fit.rows_per_second,
            "process_peak_rss_bytes": fit.process_peak_rss_bytes,
        },
        "reports": [item.model_dump(mode="python") for item in reports],
    }


def finalize_attention_artifact(
    workspace: Path,
    destination: Path,
    *,
    schema: AttentionCardSchema,
    protocol: AttentionProtocol,
    fit: AttentionFitResult,
    reports: Sequence[AttentionEvaluationReport],
    model_version: str,
    chunk_size: int = 8 * 1024 * 1024,
) -> Path:
    """Write checked model members, then rename the owned workspace once."""
    if chunk_size < 1 or not workspace.is_dir() or destination.exists():
        raise AttentionArtifactError("artifact workspace or destination is invalid")
    if (
        schema.fingerprint() != protocol.encoding_sha256
        or fit.refit_model.schema_fingerprint != schema.fingerprint()
        or fit.config.seed != fit.runtime.seed
    ):
        raise AttentionArtifactError("artifact fit, protocol, and schema disagree")
    expected_roles = {"development"}
    if protocol.reporting is not None:
        expected_roles.add("reporting")
    if {item.role for item in reports} != expected_roles or len(reports) != len(expected_roles):
        raise AttentionArtifactError("artifact evaluation roles are incomplete")
    if any(item.protocol != protocol or item.fit_artifact_id != model_version for item in reports):
        raise AttentionArtifactError("artifact evaluations do not match fit identity")
    (workspace / "card-catalog.json").write_bytes(canonical_json_bytes(schema.catalog_snapshot))
    (workspace / "feature-schema.json").write_bytes(
        canonical_json_bytes(schema.model_dump(mode="python")) + b"\n"
    )
    (workspace / "evaluation.json").write_bytes(
        canonical_json_bytes(_evaluation_payload(fit, reports)) + b"\n"
    )
    weights = {
        name: value.detach().to("cpu").clone()
        for name, value in fit.refit_model.state_dict().items()
    }
    torch.save(weights, workspace / "weights.pt")
    manifest = AttentionArtifactManifest(
        probability_interpretation=schema.probability_interpretation,
        model_version=model_version,
        dataset_version=protocol.dataset_version,
        catalog_version=schema.catalog_version,
        balance_era_id=protocol.balance_era_id,
        encoding_sha256=schema.fingerprint(),
        fit_protocol=protocol,
        fit_seed=fit.config.seed,
        runtime=AttentionArtifactRuntime(
            device=fit.runtime.device,
            python_version=fit.runtime.python_version,
            torch_version=fit.runtime.torch_version,
        ),
        files=_inventory(workspace, chunk_size),
    )
    (workspace / "manifest.json").write_bytes(dump_attention_manifest(manifest))
    load_attention_artifact(workspace, chunk_size=chunk_size)
    return publish_processed_version(workspace, destination)


def _load_weights(path: Path, model: AttentionMatchupModel) -> None:
    """Accept precisely the reconstructed model's finite CPU tensor state."""
    try:
        raw: object = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, UnpicklingError, ValueError) as error:
        raise AttentionArtifactError("attention weights cannot be loaded") from error
    if not isinstance(raw, dict):
        raise AttentionArtifactError("attention weights must be a tensor state dict")
    raw_dict = cast(dict[object, object], raw)
    if any(not isinstance(key, str) for key in raw_dict):
        raise AttentionArtifactError("attention weights must be a tensor state dict")
    state = cast(dict[str, object], raw_dict)
    expected = model.state_dict()
    if state.keys() != expected.keys():
        raise AttentionArtifactError("attention weight parameter names do not match schema")
    for name, value in state.items():
        reference = expected[name]
        if (
            not isinstance(value, Tensor)
            or value.shape != reference.shape
            or value.dtype != reference.dtype
            or value.device.type != "cpu"
            or not torch.isfinite(value).all().item()
        ):
            raise AttentionArtifactError(f"attention weight tensor is invalid: {name}")
    model.load_state_dict(cast(dict[str, Tensor], state), strict=True)
    model.eval()


def load_attention_artifact(
    artifact: Path, *, chunk_size: int = 8 * 1024 * 1024
) -> tuple[AttentionArtifactManifest, AttentionCardSchema, AttentionMatchupModel]:
    """Verify every inventoried byte and return an inference-ready CPU model."""
    if chunk_size < 1 or not (artifact / "manifest.json").is_file():
        raise AttentionArtifactError("attention artifact manifest is required")
    try:
        manifest = AttentionArtifactManifest.model_validate_json(
            (artifact / "manifest.json").read_bytes()
        )
        actual = _inventory(artifact, chunk_size)
        if actual != manifest.files:
            raise AttentionArtifactError("attention artifact file inventory mismatch")
        schema = AttentionCardSchema.model_validate_json(
            (artifact / "feature-schema.json").read_bytes()
        )
        if (
            schema.fingerprint() != manifest.encoding_sha256
            or schema.catalog_version != manifest.catalog_version
            or schema.balance_era_id != manifest.balance_era_id
            or schema.probability_interpretation != manifest.probability_interpretation
            or (artifact / "card-catalog.json").read_bytes()
            != canonical_json_bytes(schema.catalog_snapshot)
        ):
            raise AttentionArtifactError("attention artifact schema or catalog mismatch")
        evaluation: object = loads((artifact / "evaluation.json").read_bytes())
        if not isinstance(evaluation, dict):
            raise AttentionArtifactError("attention evaluation is invalid")
        evaluation_body = cast(dict[str, object], evaluation)
        fit_body = evaluation_body.get("fit")
        report_values = evaluation_body.get("reports")
        if not isinstance(fit_body, dict) or not isinstance(report_values, list):
            raise AttentionArtifactError("attention evaluation is invalid")
        fit_values = cast(dict[str, object], fit_body)
        fit_config = AttentionFitConfig.model_validate(fit_values.get("config"))
        selected = fit_values.get("selected_epoch")
        history = fit_values.get("history")
        if (
            fit_config.seed != manifest.fit_seed
            or fit_config.device != manifest.runtime.device
            or type(selected) is not int
            or not 1 <= selected <= fit_config.max_epochs
            or not isinstance(history, list)
            or len(cast(list[object], history)) < selected
        ):
            raise AttentionArtifactError("attention fit provenance is invalid")
        reports = tuple(
            AttentionEvaluationReport.model_validate(item)
            for item in cast(list[object], report_values)
        )
        expected_roles = {"development"}
        if manifest.fit_protocol.reporting is not None:
            expected_roles.add("reporting")
        if (
            any(
                item.protocol != manifest.fit_protocol
                or item.fit_artifact_id != manifest.model_version
                for item in reports
            )
            or {item.role for item in reports} != expected_roles
            or len(reports) != len(expected_roles)
        ):
            raise AttentionArtifactError("attention evaluation provenance is invalid")
        model = AttentionMatchupModel(schema)
        _load_weights(artifact / "weights.pt", model)
        return manifest, schema, model
    except (OSError, ValueError, ValidationError) as error:
        if isinstance(error, AttentionArtifactError):
            raise
        raise AttentionArtifactError("attention artifact is invalid") from error
