"""Controlled-run eligibility, exact evaluation-role inventories, and fixed covariates."""

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from clash_sos.domain.attention_protocol import AttentionProtocol, RowKey, digest_row_keys
from clash_sos.domain.canonical_dataset import canonical_json_bytes
from clash_sos.domain.manifests import ManifestModel
from experiments.common.artifacts import file_record, load_run
from experiments.common.checkpoints import Checkpoint, load_checkpoint
from experiments.common.contracts import RunManifest, StudyConfig, Variant, fingerprint
from experiments.common.ensembles import SeedRun
from experiments.common.predictions import Prediction, read_predictions
from experiments.common.slices import FixedSlices, SliceMetadata, SliceSpec, build_slices


class RefitCovariate(ManifestModel):
    key: RowKey
    event_key: str
    player_a: str
    player_b: str
    tokens: tuple[tuple[int, ...], tuple[int, ...]]


class SupportInventory(ManifestModel):
    protocol: AttentionProtocol
    refit: tuple[RefitCovariate, ...]
    development: tuple[RefitCovariate, ...]


@dataclass(frozen=True)
class ComparisonRun:
    manifest: RunManifest
    checkpoint: Checkpoint
    calibration: tuple[Prediction, ...]
    development: tuple[Prediction, ...]
    support: SupportInventory
    asset: SeedRun


def load_comparison_run(
    directory: Path,
    config: StudyConfig,
    variant: Variant,
    seed: int,
    penalty: float,
    *,
    shared: bool,
) -> ComparisonRun:
    manifest = load_run(directory)
    if (
        not manifest.eligible_for_comparison
        or manifest.status != "complete"
        or manifest.dirty_sha256
    ):
        raise ValueError("comparison requires complete clean eligible runs")
    if manifest.stage != "development-frozen" or manifest.population != config.population:
        raise ValueError("comparison requires identical frozen population")
    if any(member not in manifest.inputs for member in manifest.population.snapshot_files):
        raise ValueError("comparison source inventory does not match the frozen population")
    if shared:
        if manifest.config.study_id != "response-cycle" or (
            manifest.config.optimizer != config.optimizer or manifest.config.seeds != config.seeds
        ):
            raise ValueError("shared baseline requires identical optimizer and seed registry")
    elif manifest.config_sha256 != fingerprint(config):
        raise ValueError("comparison configuration hash mismatch")
    runtime = dict(manifest.runtime)
    expected_id = "A0" if shared else variant.variant_id
    phase = (
        "reference"
        if variant.architecture == "attention"
        else "screen"
        if seed == 0
        else "confirmation"
    )
    if len(runtime) != len(manifest.runtime) or (
        runtime.get("variant_id"),
        runtime.get("seed"),
        runtime.get("penalty"),
        runtime.get("phase"),
    ) != (expected_id, str(seed), str(float(penalty)), phase):
        raise ValueError("comparison run settings do not match registered seed and penalty")
    checkpoint = load_checkpoint(directory / "checkpoint.pt")
    if checkpoint.metadata.variant.model_dump(exclude={"variant_id"}) != variant.model_dump(
        exclude={"variant_id"}
    ):
        raise ValueError("checkpoint variant differs from the frozen comparison registry")
    members = {member.path: member for member in manifest.outputs}
    required = {"checkpoint.pt", "calibration.json", "development.json", "support.json"}
    if not required <= members.keys():
        raise ValueError("comparison requires inventoried checkpoint, predictions, and support")
    support = SupportInventory.model_validate_json((directory / "support.json").read_bytes())
    if support.protocol.reporting is not None:
        raise ValueError("comparison support cannot contain a reporting role")
    if support.protocol.calibration is None:
        raise ValueError("comparison requires a declared calibration role")
    if (
        len(support.refit) != support.protocol.refit.row_count
        or digest_row_keys(row.key for row in support.refit)
        != support.protocol.refit.row_keys_sha256
    ):
        raise ValueError("comparison refit covariates do not match the frozen support role")
    if len({row.event_key for row in support.refit}) != len(support.refit):
        raise ValueError("comparison refit requires unique event identities")
    calibration, development = (
        read_predictions(directory / name) for name in ("calibration.json", "development.json")
    )
    for rows, role in (
        (calibration, support.protocol.calibration),
        (development, support.protocol.development),
    ):
        if (
            len(rows) != role.row_count
            or digest_row_keys(row.row_key for row in rows) != role.row_keys_sha256
        ):
            raise ValueError("prediction inventory does not match its frozen evaluation role")
        if any(not role.start <= row.timestamp < role.end for row in rows):
            raise ValueError("prediction timestamps exceed their frozen evaluation role")
    inventory = sorted(
        [
            *((row.key, row.event_key) for row in support.refit),
            *((row.row_key, row.event_key) for row in (*calibration, *development)),
        ]
    )
    if digest_row_keys(key for key, _ in inventory) != manifest.population.row_keys_sha256 or (
        sha256(canonical_json_bytes(tuple(inventory))).hexdigest()
        != manifest.population.event_mapping_sha256
    ):
        raise ValueError("comparison roles must reconstruct the exact frozen event population")
    if any(
        not manifest.population.start <= key[0] < manifest.population.end for key, _ in inventory
    ):
        raise ValueError("comparison roles exceed the frozen population bounds")
    joined = {row.key: row for row in support.development}
    if len(joined) != len(support.development) or set(joined) != {
        row.row_key for row in development
    }:
        raise ValueError("development covariates must join one-to-one to predictions")
    for row in development:
        covariates = joined[row.row_key]
        if (row.event_key, row.player_a, row.player_b) != (
            covariates.event_key,
            covariates.player_a,
            covariates.player_b,
        ):
            raise ValueError("development covariate event or player orientation mismatch")
    asset = SeedRun(
        seed=seed,
        directory=str(directory.resolve()),
        manifest_sha256=file_record(directory / "manifest.json", directory).sha256,
        checkpoint_sha256=members["checkpoint.pt"].sha256,
        calibration_sha256=members["calibration.json"].sha256,
        development_sha256=members["development.json"].sha256,
        run_config_sha256=manifest.config_sha256,
    )
    return ComparisonRun(manifest, checkpoint, calibration, development, support, asset)


def slice_metadata(
    comparator: ComparisonRun,
) -> tuple[SliceMetadata, ...]:
    refit = comparator.support.refit
    lineups: Counter[tuple[int, ...]] = Counter(side for row in refit for side in row.tokens)
    pairs: Counter[tuple[tuple[int, ...], tuple[int, ...]]] = Counter(
        (min(row.tokens), max(row.tokens)) for row in refit
    )
    players: Counter[str] = Counter(
        player for row in refit for player in (row.player_a, row.player_b)
    )
    schema = comparator.checkpoint.metadata.input_schema
    catalog = comparator.checkpoint.catalog
    return tuple(
        SliceMetadata(
            row_key=row.key,
            era=schema.balance_era_id,
            deck_a_support=lineups[row.tokens[0]],
            deck_b_support=lineups[row.tokens[1]],
            pair_support=pairs[(min(row.tokens), max(row.tokens))],
            player_a_support=players[row.player_a],
            player_b_support=players[row.player_b],
            history_a_count=players[row.player_a],
            history_b_count=players[row.player_b],
            forms=tuple(
                sorted(
                    {
                        catalog.for_token(token).identity
                        for side in row.tokens
                        for token in side[:8]
                        if not catalog.for_token(token).identity.endswith(":base")
                    }
                )
            ),
            tower_a=schema.identity_vocab[row.tokens[0][-1]],
            tower_b=schema.identity_vocab[row.tokens[1][-1]],
        )
        for row in comparator.support.development
    )


def fixed_slices(reference: ComparisonRun, comparator: tuple[Prediction, ...]) -> FixedSlices:
    metadata = slice_metadata(reference)
    schema = reference.checkpoint.metadata.input_schema
    return build_slices(
        metadata,
        comparator,
        SliceSpec(
            days=tuple(sorted({row.row_key[0].date() for row in metadata})),
            eras=(schema.balance_era_id,),
            forms=tuple(
                identity
                for identity in schema.identity_vocab
                if identity.rsplit(":", 1)[1] in ("evolution", "hero", "champion")
            ),
            towers=tuple(schema.identity_vocab[token] for token in sorted(schema.tower_indices)),
        ),
    )
