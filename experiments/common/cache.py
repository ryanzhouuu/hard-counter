from hashlib import sha256
from pathlib import Path

import numpy as np
from pydantic import Field

from clash_sos.domain.attention_protocol import digest_row_keys
from clash_sos.domain.canonical_dataset import canonical_json_bytes
from clash_sos.domain.manifests import ManifestModel, Sha256
from experiments.common.artifacts import file_record, verify_files
from experiments.common.contracts import FileRecord
from experiments.common.data_access import ResearchRow


class FeatureCacheIdentity(ManifestModel):
    oriented_sha256: Sha256
    encoding_sha256: Sha256
    mechanics_sha256: Sha256
    feature_names: tuple[str, ...]
    formulas: tuple[str, ...]
    training_rows_sha256: Sha256
    scales: tuple[float, ...]
    row_keys_sha256: Sha256
    row_count: int = Field(gt=0)


def oriented_digest(rows: tuple[ResearchRow, ...]) -> str:
    payload = [
        (r.key, r.event_key, r.tokens, r.player_a, r.player_b, r.label, r.mirrored) for r in rows
    ]
    return sha256(canonical_json_bytes(payload)).hexdigest()


def write_feature_cache(
    directory: Path,
    rows: tuple[ResearchRow, ...],
    values: np.ndarray,
    identity: FeatureCacheIdentity,
) -> FileRecord:
    if identity.row_keys_sha256 != digest_row_keys(r.key for r in rows):
        raise ValueError("feature row digest mismatch")
    if identity.oriented_sha256 != oriented_digest(rows):
        raise ValueError("feature orientation mismatch")
    if values.shape != (len(rows), len(identity.feature_names)) or not np.isfinite(values).all():
        raise ValueError("feature layout mismatch")
    if len(identity.formulas) != values.shape[1] or len(identity.scales) != values.shape[1]:
        raise ValueError("feature definitions/scales mismatch")
    if identity.row_count != len(rows) or len(set(identity.feature_names)) != values.shape[1]:
        raise ValueError("feature count/order mismatch")
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "features.npy").open("xb") as out:
        np.save(out, values, allow_pickle=False)
    member = file_record(directory / "features.npy", directory, row_count=len(rows))
    (directory / "identity.json").write_bytes(canonical_json_bytes(identity.model_dump()))
    (directory / "member.json").write_bytes(canonical_json_bytes(member.model_dump()))
    return member


def load_feature_cache(directory: Path, expected: FeatureCacheIdentity) -> np.ndarray:
    identity = FeatureCacheIdentity.model_validate_json((directory / "identity.json").read_bytes())
    if identity != expected:
        raise ValueError("feature cache configuration mismatch")
    member = FileRecord.model_validate_json((directory / "member.json").read_bytes())
    verify_files(directory, (member,))
    values = np.load(directory / member.path, mmap_mode="r", allow_pickle=False)
    if values.shape != (identity.row_count, len(identity.feature_names)):
        raise ValueError("feature cache shape mismatch")
    return values
