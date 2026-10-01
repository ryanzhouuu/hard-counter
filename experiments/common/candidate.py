"""Explicit candidate freezes and future population compatibility checks."""

from collections.abc import Mapping
from datetime import datetime
from typing import Self

from pydantic import Field, model_validator

from clash_sos.domain.manifests import ManifestModel, Sha256
from experiments.common.contracts import (
    ComparisonRules,
    FileRecord,
    PopulationIdentity,
    StudyConfig,
    fingerprint,
)
from experiments.common.matrix import screen_jobs


class CandidateSelection(ManifestModel):
    variant_id: str = Field(min_length=1)
    penalty: float = Field(gt=0, allow_inf_nan=False)
    seeds: tuple[int, ...]


class CandidateFreeze(ManifestModel):
    source_config_sha256: Sha256
    source_configs: tuple[tuple[str, Sha256], ...] = ()
    fit_population: PopulationIdentity
    selections: tuple[CandidateSelection, ...]
    confirmation_rules: ComparisonRules
    comparison_ids: tuple[str, ...]
    code_sha256: Sha256
    assets: tuple[FileRecord, ...]
    frozen_at: datetime
    inspected_through: datetime
    reporting_start: datetime

    @model_validator(mode="after")
    def validate_boundaries(self) -> Self:
        if any(
            stamp.tzinfo is None or stamp.utcoffset() is None
            for stamp in (self.frozen_at, self.inspected_through, self.reporting_start)
        ):
            raise ValueError("freeze timestamps must have timezones")
        if self.frozen_at <= self.inspected_through or self.reporting_start <= max(
            self.frozen_at, self.inspected_through, self.fit_population.end
        ):
            raise ValueError(
                "reporting must begin strictly after development, inspection, and freeze"
            )
        if not self.assets or len({member.path for member in self.assets}) != len(self.assets):
            raise ValueError("candidate assets must be nonempty and uniquely inventoried")
        if len(dict(self.source_configs)) != len(self.source_configs):
            raise ValueError("candidate source configuration identities must be unique")
        ids = [entry.variant_id for entry in self.selections]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("freeze requires unique selected variants")
        if any(
            not entry.seeds or len(set(entry.seeds)) != len(entry.seeds) or min(entry.seeds) < 0
            for entry in self.selections
        ):
            raise ValueError("candidate seeds must be nonempty unique nonnegative values")
        if (
            self.confirmation_rules.confirmation is None
            or self.confirmation_rules.test_alternative is None
        ):
            raise ValueError("candidate freeze requires preregistered confirmation rules")
        family = set(self.comparison_ids)
        if len(family) != len(self.comparison_ids) or family not in (
            set(ids),
            {*ids, "full_attention"},
        ):
            raise ValueError(
                "comparison family must contain exactly selections and optional attention"
            )
        if self.confirmation_rules.confirmation == "single" and len(family) != 1:
            raise ValueError("single confirmation requires exactly one comparison")
        return self


def freeze_candidate(
    config: StudyConfig,
    selected_penalties: Mapping[str, float],
    *,
    code_sha256: str,
    assets: tuple[FileRecord, ...],
    frozen_at: datetime,
    inspected_through: datetime,
    reporting_start: datetime,
    confirmation_rules: ComparisonRules,
    include_attention: bool = False,
) -> CandidateFreeze:
    """Record a caller's explicit decision without changing the source run stage."""
    if config.stage != "development-frozen" or config.population is None:
        raise ValueError("candidate freeze requires a completed development design")
    registered = {job.variant_id for job in screen_jobs(config) if job.penalty > 0}
    if (
        not selected_penalties
        or not set(selected_penalties) <= registered
        or any(penalty not in config.penalties for penalty in selected_penalties.values())
    ):
        raise ValueError("candidate selection must use enabled variants and registered penalties")
    if include_attention and not any(
        variant.enabled
        and variant.architecture == "attention"
        and variant.variant_id == "full_attention"
        for variant in config.variants
    ):
        raise ValueError("attention confirmation requires a registered enabled reference")
    return CandidateFreeze(
        source_config_sha256=fingerprint(config),
        fit_population=config.population,
        selections=tuple(
            CandidateSelection(variant_id=variant, penalty=penalty, seeds=config.seeds)
            for variant, penalty in sorted(selected_penalties.items())
        ),
        confirmation_rules=confirmation_rules,
        comparison_ids=tuple(
            sorted((*selected_penalties, *(("full_attention",) if include_attention else ())))
        ),
        code_sha256=code_sha256,
        assets=assets,
        frozen_at=frozen_at,
        inspected_through=inspected_through,
        reporting_start=reporting_start,
    )


def validate_candidate_population(frozen: CandidateFreeze, population: PopulationIdentity) -> None:
    if population.start < frozen.reporting_start:
        raise ValueError("backfilled battles cannot enter prospective reporting")
    original = frozen.fit_population
    if (population.mode, population.level, population.era, population.catalog_sha256) != (
        original.mode,
        original.level,
        original.era,
        original.catalog_sha256,
    ):
        raise ValueError(
            "prospective population has an incompatible mode, level, era, or vocabulary"
        )
    if {item.sha256 for item in original.snapshot_files} & {
        item.sha256 for item in population.snapshot_files
    }:
        raise ValueError("prospective reporting requires physically separate source inventories")
