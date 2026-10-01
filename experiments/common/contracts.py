from datetime import UTC, datetime
from hashlib import sha256
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from clash_sos.domain.canonical_dataset import canonical_json_bytes
from clash_sos.domain.manifests import ManifestModel, Sha256, validate_relative_path

Stage = Literal[
    "preparation/smoke",
    "development-frozen",
    "candidate-frozen",
    "prospective-reporting",
    "integration-review",
]
Role = Literal["selection_fit", "watch", "refit", "calibration", "development", "reporting"]


def fingerprint(value: ManifestModel) -> str:
    return sha256(canonical_json_bytes(value.model_dump(mode="python"))).hexdigest()


class FileRecord(ManifestModel):
    path: str
    sha256: Sha256
    size_bytes: int = Field(ge=0)
    row_count: int | None = Field(default=None, ge=0)

    @field_validator("path")
    @classmethod
    def relative(cls, value: str) -> str:
        validate_relative_path(value)
        return value


class PopulationIdentity(ManifestModel):
    snapshot_files: tuple[FileRecord, ...]
    row_keys_sha256: Sha256
    event_mapping_sha256: Sha256
    oriented_sha256: Sha256
    mode: str = Field(min_length=1)
    level: Literal[16] = 16
    era: str = Field(min_length=1)
    mirror_seed: int = Field(ge=0)
    start: datetime
    end: datetime
    catalog_sha256: Sha256

    @field_validator("start", "end")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("population bounds must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def bounds(self) -> Self:
        paths = tuple(item.path for item in self.snapshot_files)
        if self.start >= self.end or not paths or len(set(paths)) != len(paths):
            raise ValueError("population requires ordered bounds and unique source files")
        return self


class Variant(ManifestModel):
    variant_id: str = Field(min_length=1)
    feature_groups: tuple[str, ...] = ()
    nuisance: Literal["none", "history", "joint"] = "none"
    architecture: Literal["explicit", "attention"] = "explicit"
    schema_sha256: Sha256 | None = None
    feature_sha256: Sha256 | None = None
    feature_names: tuple[str, ...] = ()
    enabled: bool = True
    disabled_reason: str | None = None

    @model_validator(mode="after")
    def gate(self) -> Self:
        if self.enabled == bool(self.disabled_reason):
            raise ValueError("disabled variants require a reason; enabled variants cannot have one")
        if self.nuisance != "none" and self.feature_groups:
            raise ValueError("independent studies cannot combine nuisance and mechanics features")
        return self


class ComparisonRules(ManifestModel):
    seed_aggregation: Literal["mean_probability"] = "mean_probability"
    primary: Literal["paired_calibrated_log_loss"] = "paired_calibrated_log_loss"
    minimum_slice_rows: int = Field(default=30, gt=0)
    minimum_pattern_support: int = Field(default=2, gt=0)
    practical_margin: float | None = Field(default=None, gt=0)
    maximum_brier_regression: float | None = Field(default=None, ge=0)
    maximum_daily_regression: float | None = Field(default=None, ge=0)
    confirmation: Literal["single", "holm"] | None = None
    test_alternative: Literal["two_sided", "improvement"] | None = None


class OptimizerConfig(ManifestModel):
    batch_size: int = Field(default=256, gt=0)
    max_epochs: int = Field(default=20, gt=0)
    patience: int = Field(default=3, gt=0)
    learning_rate: float = Field(default=0.001, gt=0, allow_inf_nan=False)
    weight_decay: float = Field(default=0.0001, ge=0, allow_inf_nan=False)
    gradient_clip_norm: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    device: str = "cpu"
    time_limit_seconds: float = Field(default=300.0, gt=0, allow_inf_nan=False)


class StudyConfig(ManifestModel):
    study_id: str = Field(min_length=1)
    version: int = Field(default=1, gt=0)
    objective: Literal["equal-skill-deck-and-tower"] = "equal-skill-deck-and-tower"
    stage: Stage = "preparation/smoke"
    variants: tuple[Variant, ...]
    comparator_ids: tuple[str, ...] = ("explicit",)
    seeds: tuple[int, ...] = (0, 1, 2)
    penalties: tuple[float, ...] = (0.0001, 0.001, 0.01)
    optimizer: OptimizerConfig = OptimizerConfig()
    calibration: Literal["positive_temperature"] = "positive_temperature"
    feature_order: tuple[str, ...] = ()
    feature_definitions_sha256: Sha256 | None = None
    schema_sha256: Sha256 | None = None
    attributes_sha256: Sha256 | None = None
    mechanics_sha256: Sha256 | None = None
    population: PopulationIdentity | None = None
    prospective: PopulationIdentity | None = None
    decision_sha256: Sha256 | None = None
    smoke_row_cap: int = Field(default=128, ge=12)
    smoke_epoch_cap: int = Field(default=8, gt=0)
    smoke_time_cap: float = Field(default=30.0, gt=0)
    slices: tuple[str, ...] = ("day", "tower", "form", "player_novelty", "refit_support")
    rules: ComparisonRules = ComparisonRules()

    @model_validator(mode="after")
    def registry(self) -> Self:
        ids = tuple(v.variant_id for v in self.variants)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("variants must have unique IDs")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or min(self.seeds) < 0:
            raise ValueError("seeds must be unique nonnegative integers")
        if not self.penalties or any(not 0 < p < float("inf") for p in self.penalties):
            raise ValueError("penalties must be positive finite values")
        if len(set(self.feature_order)) != len(self.feature_order):
            raise ValueError("feature order must be unique")
        if self.stage != "preparation/smoke":
            if self.population is None or self.decision_sha256 is None:
                raise ValueError("advancement requires explicit population and versioned decision")
            if self.schema_sha256 is None or self.feature_definitions_sha256 is None:
                raise ValueError("frozen stages require schema and feature definitions")
            if self.rules.practical_margin is None or self.rules.maximum_brier_regression is None:
                raise ValueError("frozen stages require comparison thresholds")
        if self.stage == "prospective-reporting" and (
            self.prospective is None
            or self.rules.confirmation is None
            or self.rules.test_alternative is None
        ):
            raise ValueError("reporting requires prospective population and confirmation design")
        return self


class RunManifest(ManifestModel):
    run_id: str = Field(min_length=1)
    config: StudyConfig
    config_sha256: Sha256
    git_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    dirty_sha256: Sha256 | None = None
    lock_sha256: Sha256
    stage: Stage
    status: Literal["complete", "failed", "time_limited"]
    eligible_for_comparison: bool
    inputs: tuple[FileRecord, ...]
    outputs: tuple[FileRecord, ...]
    population: PopulationIdentity
    runtime: tuple[tuple[str, str], ...]
    selected_epochs: tuple[int, ...] = ()
    temperatures: tuple[float, ...] = ()
    scales: tuple[float, ...] = ()
    failures: tuple[str, ...] = ()

    @model_validator(mode="after")
    def reconcile(self) -> Self:
        if self.config_sha256 != fingerprint(self.config) or self.stage != self.config.stage:
            raise ValueError("run configuration digest or stage mismatch")
        if self.eligible_for_comparison and (
            self.status != "complete" or self.stage == "preparation/smoke" or self.dirty_sha256
        ):
            raise ValueError("incomplete, dirty, and exploratory runs cannot enter comparisons")
        if self.status != "complete" and not self.failures:
            raise ValueError("failed runs require failure details")
        for files in (self.inputs, self.outputs):
            if len({f.path for f in files}) != len(files):
                raise ValueError("file inventories require unique paths")
        return self
