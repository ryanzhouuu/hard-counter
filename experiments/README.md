# Experiments

This workspace is for reproducible ML studies: research entrypoints, study
configurations, experiment-specific models, and their tests. Import existing
backend contracts and utilities where possible. Logic shared with training or
serving belongs in `backend/src/clash_sos`.

## Repository conventions

- Track reusable code, synthetic fixtures, reviewed mechanics inputs, and study
  definitions. Keep modules small and give each study its own directory.
- Keep datasets, derived caches, checkpoints, predictions, and generated reports
  under ignored `data/experiments/` or `models/experiments/`.
- Keep local working documents under ignored `docs/.local/`.
- Experiment tests run in the normal quality checks using synthetic data,
  without local datasets or API credentials. The package is excluded from the
  production wheel.
- Record source revision, configuration, input hashes, row identities, seeds,
  runtime, and artifact hashes with every completed run. Refuse to overwrite
  completed outputs.

## Evaluation conventions

Preparation and smoke runs validate correctness, reproducibility, and resource
use. Model comparisons require a frozen dataset and evaluation protocol.
Separate fitting, epoch selection, calibration, development comparison, and
prospective reporting. Search entrypoints must exclude reporting labels.

Compare variants on identical battles and preserve deck-order invariance and
side-swap probability complementarity. Report scoring coverage and repeated-player
uncertainty alongside probability metrics. Observed battle outcomes alone do not
establish that an equal-skill interpretation is identified.

Experimental checkpoints do not change the selected live model. Production
integration follows a completed comparison and compatibility checks.

## Running the studies

Run commands from the repository root with the ML extra installed:

```sh
uv sync --extra ml --dev
uv run --extra ml python -m experiments --synthetic
```

The second command runs every enabled branch in all five studies, including the
native attention reference. Each study also has an independent entrypoint:

```sh
uv run --extra ml python -m experiments.matchup_features.study smoke --synthetic
uv run --extra ml python -m experiments.player_adjustment.study smoke --synthetic
uv run --extra ml python -m experiments.form_mechanics.study smoke --synthetic
uv run --extra ml python -m experiments.tower_mechanics.study smoke --synthetic
uv run --extra ml python -m experiments.higher_order.study smoke --synthetic
```

All entrypoints accept `validate`, `smoke`, `run-development`, `compare`, and
`confirm`; use `--help` for their shared options. `validate` emits covariate,
mechanics-coverage, and variant-registry information without scoring outcomes.
Reusable definitions are in [configs](configs/). Smoke defaults are 128 rows,
eight epochs per selection/refit stage, and 30 seconds per fit. Fits run serially.
Use `--row-cap` to declare a different bounded input budget. Synthetic populations
require a multiple of 32, at least 64 rows.

Each run has a unique `--run-id`. Verified checkpoints, feature caches,
predictions, support inventories, and manifests are co-located under
`models/experiments/<study>/<run-id>/`; summaries, source inventories, readiness,
and comparison reports go under `data/experiments/<study>/<run-id>/`. `--models`
and `--output` override these roots. Failed or time-limited fits retain a failure
record and remain ineligible for comparisons. A completed output cannot be
overwritten; exact resumes verify configuration, code, inputs, and output hashes.

Reports include time per epoch, throughput, peak process memory, output bytes,
finite gradients, swap error, and calibration status. Smoke metrics describe
execution and correctness; they cannot select penalties, promote a stage, or
support an improvement claim.

## Replacing synthetic inputs

Use an immutable official snapshot built by the existing dataset preparation
pipeline. Keep its nonempty physical train, validation, and test partitions.
Create a search protocol with
`experiments.common.protocol.search_protocol(original, dataset, calibration_end)`:
this splits validation into calibration then development, keeps equal timestamps
together, and removes reporting. The physical test partition is verified for
integrity but is not materialized or scored by search. Snapshot data already
inspected are development evidence, even when stored in a file named test.

Save the search protocol and its matching explicit-only `AttentionCardSchema`
under an ignored input directory, and supply all four paths:

```sh
uv run --extra ml python -m experiments.player_adjustment.study validate \
  --dataset data/processed/your-snapshot \
  --protocol data/experiments/inputs/search-protocol.json \
  --schema data/experiments/inputs/explicit-schema.json \
  --cache data/experiments/inputs/cache --row-cap 2048
```

Replace `validate` with `smoke` for bounded exploratory fitting. The cap is a
ceiling: exceeding it fails before feature construction; rows are never silently
truncated. The adapter verifies canonical row/event joins, labels, players, and
joint mirroring against the native cache. Attention runs create their own
network-specific schema/protocol/cache and verify identical oriented populations.
Do not use the production training command to search research variants.

The default [mechanics input](mechanics/inputs/2026-10-02-partial-r8.json) records
field-level evidence and explicit unknowns. The [mechanics review](../docs/MECHANICS_REVIEW.md)
documents corrected capabilities, original release-preview sources, and remaining
coverage limits. September inputs remain frozen. Supply `--mechanics` for another
reviewed catalog. Its applicability must cover the population for mechanics
variants. Unknown categorical fields have registered missingness representations;
quantitative tower descriptors and predicates requiring verified values fail with
an explicit unavailable reason. Synthetic completeness is not evidence of real
catalog completeness. Mirror, conditional costs, form activation, and tower
deploy-cost inapplicability have separate representations.

Synthetic mechanics are accepted only with synthetic preparation fixtures;
real snapshot adapters and controlled fits reject them. Response missingness
counts unknown attacking threat fields and opposing answer predicates separately,
so unknown card capabilities survive side-swap antisymmetrization.

An unknown spell classification does not establish a deployable response. Tower
answers use their separate kind and never receive a deployment cost. Before an
air-only comparison, call `mechanics.air_audit.require_air_mechanics` on the
population's bound tokens. `air_audit` lists incomplete identities and fields;
`include_cost=True` adds the separate deployment-cost gate. This preparation
check covers ordinary channels; ability-only and summoned-unit channels need
separate verification. Existing exploratory response extraction retains its
missingness features and does not automatically enforce this strict gate.

Manual facts use `user_reported` status and count as known mechanics. Their
dated statements are saved in `mechanics/evidence/` and bound by SHA-256; they
need no publisher URL. Loading verifies the referenced report and exact field
claims. This status distinguishes manual observations from linked sources without
discarding either. Optional evidence fields are omitted from older snapshots'
serialization, preserving their existing digests.

The fourth October revision covers ordinary air predicates for 185 identities,
using primary sources and explicitly documented community Wiki evidence. Wiki
field dates record review on October 1, 2026. Earlier snapshots and manual tower/
Freeze evidence remain frozen. Confirmed spells have no primary airborne actor.
Spirit Empress retains unknown unconditional flags: `airborne_condition` and
`targets_air_condition` record `available_elixir_at_least_6`. These evidence
categories do not resolve deployment state or bypass the strict air gate.

The fifth revision explicitly classifies Spirit Empress with the optional
`airborne_mode: hybrid` category. Allowed modes are `ground`, `air`, and `hybrid`;
hybrid means deployment can produce either form. Her sourced conditions still
describe which form appears, and the static booleans retain their existing meaning.

The sixth revision adds potential conditional air channels for Spirit Empress,
Hero Wizard, Hero Ice Golem, Hero Giant, Mighty Miner, and Monk. The flags
`conditional_airborne`, `conditional_air_damage`, `conditional_air_control`, and
`conditional_air_reflection` describe capability, never observed activation.
`conditional_air_trigger`, `conditional_air_response_scope`, and
`conditional_air_control_kind` retain trigger, eligible targets, and effect type;
`ability_usage` records single-use per deployment separately from `ability_cost`.
Only reviewed effects are populated; omitted effects stay unknown. Ordinary flags
and the strict static audit retain their meaning. Feature extraction consumes
the new fields when the `conditional_air` feature group is enabled. See the
mechanics review for evidence and category definitions.

Use `air_audit(..., include_conditional=True)` or
`require_air_mechanics(..., include_conditional=True)` to audit ordinary
predicates plus recorded positive conditional channels. This mode accepts a
documented six-Elixir hybrid only with matching conditions, flight/damage
capabilities, non-spell classification, and the flying-form response scope.
It checks trigger, scope, control type, and paid-ability usage metadata;
`include_cost=True` also requires paid-ability cost independently of deployment
cost, and known ordinary hand-cycle applicability. The eighth catalog revision
completes these fields for all 182 card/form identities; Mirror and Spirit
Empress retain conditional costs and are excluded from fixed-cost return proxies.
The default strict static gate continues to require ordinary booleans.
Unreviewed conditional capabilities remain coverage unknowns: passing the
conditional-aware gate does not certify an exhaustive ability inventory.

Response/cycle variant A4 enables `response`, `cycle`, and `conditional_air`;
A3 retains the ordinary response/cycle registry for comparison. The new group
registers separate potential flight, damage, control, and reflection counts,
unavailable-channel counts, trigger/scope/control categories, ability-cost
summaries, and interactions with opposing ordinary/conditional air threats.
Additional response cards count each card once and exclude existing ordinary
air answers such as Hero Wizard. Giant's selected-troop restriction and Monk's
projectile eligibility remain distinct categories. These counts describe
potential channels, not defensive success or observed activation.

`validate` and smoke readiness reports include `mechanics.air_audits` for strict
static, conditional-aware, and conditional-aware plus cost gates. Unreviewed
conditional capabilities remain visible in field coverage and feature missingness.
Compatibility failures make these readiness statuses unavailable even when the
reported air-field gaps are empty. Rebuild feature caches and freeze new feature
names/hashes before controlled comparisons; historical checkpoints retain their
original feature-group definitions.

## Frozen development and candidate reporting

The checked-in configurations are preparation definitions. Before full matrices,
create a versioned configuration under an ignored run input directory. Set
`stage` to `development-frozen`, bind `population`, `decision_sha256`, schema and
feature-definition hashes, mechanics provenance, and each enabled variant's
`schema_sha256`, `feature_sha256`, and ordered `feature_names`. `validate` provides
the per-variant registry. Register practical/Brier margins, daily-regression
limits, slice/pattern support, cutoffs, seeds, optimizer budget, and any disabled
branches before scoring candidate outcomes. No default row count certifies data
readiness or effective independent support.

Run `run-development --config <frozen-config> ...`, then `compare` with the same
inputs and run ID. Screening uses seed zero and the fixed three-penalty grid;
`compare` records one development penalty decision before running seeds one and
two. A resume verifies that decision and its screening assets without selecting
again. Comparison reports and ensemble assets are published once.

Run response/cycle first. Other studies reuse its A0 runs under the same run ID,
population, optimizer, seed registry, and code revision. The new-fit budgets are
A: 26, B: 10, C: 10, D: up to 10, and E: 5 (up to 61 total); disabling the
quantitative tower branch reduces D to five. These budgets count fits containing
watch selection and deterministic restart/refit, rather than epochs.
With A4 disabled, A retains its previous 21-fit budget. A4 adds one bounded smoke
fit and five full comparison fits only after a new protocol and budget are frozen.

Comparisons retain every seed and use the same aggregation for all models:
average raw seed probabilities, convert to logits, then fit one positive
temperature on calibration only. Reports include paired log loss/Brier,
shared-player dyadic uncertainty, fixed day/tower/form/player/support slices,
and common final-10/final-25 player windows. Unsupported slices remain labeled.
Player studies calibrate matchup-only and player-aware outputs separately;
player-aware predictions are diagnostics, and joint effects remain descriptive
when deck assignment and player effects are confounded.
Player run reports include row-keyed evaluation support and missing-history flags,
support-bin counts, nuisance shrinkage, probability stability, observed-outcome
metrics/calibration, and descriptive development residuals. Controlled player
configurations must declare `player_support_bins`; the preparation configuration's
history and deck-switching edges are smoke defaults, not selected thresholds.
Player comparison reports retain these diagnostics for every selected seed and
publish separately calibrated actual-outcome ensembles, paired outcome metrics,
and matchup probability changes. Actual-output files use the
`actual-ensemble-<variant>-<role>.json` naming pattern and remain diagnostic;
penalty selection and candidate assets use matchup-only outputs.

Candidate selection is an explicit later decision. Use
`experiments.common.candidate.freeze_candidate` to record selected registered
variants/penalties, seeds, code hash, an inventory of ensemble assets, freeze time,
last inspected battle time, and a strictly later reporting start. Keep each
selected ensemble JSON and its original verified seed-run directories available.
Pass explicit `confirmation_rules` to `freeze_candidate`; these freeze the full
comparison rules, including the test alternative and multiplicity correction.
The reporting configuration must use exactly those rules. Old freezes without
these required fields must be recreated before collecting reporting outcomes.
Set `include_attention=True` to register an attention comparison in the tested
family. `single` permits exactly one comparison in total; including attention
alongside a mechanics/player challenger requires `holm`.
For a multiple-study confirmation, `CandidateFreeze.source_configs` can bind
each ensemble to its original study configuration. Include exactly one shared
baseline and the selected challengers; additional attention claims must also be
predeclared.

Prepare a physically separate future JSONL inventory of validated
`TowerBattleRowV2` rows and a `ReportingContract`; do not manufacture training
partitions. Bind its population and the candidate digest in a
`prospective-reporting` configuration. Specify `rules.confirmation` as `single`
or `holm`, and `rules.test_alternative` as `two_sided` or `improvement` before
reporting data exist. Then run:

```sh
uv run --extra ml python -m experiments.matchup_features.study confirm \
  --config data/experiments/inputs/reporting-config.json \
  --freeze data/experiments/inputs/candidate.json \
  --reporting-contract data/experiments/inputs/reporting-contract.json \
  --reporting-source data/experiments/inputs/future.jsonl --row-cap 10000
```

Confirmation verifies frozen code/assets and mode/level/era/vocabulary, rejects
backfilled or overlapping data, and uses frozen player maps, support masks,
scales, and temperatures. It verifies the dependency lock and publishes an
inventoried manifest last, including when evaluation fails. It never fits or
recalibrates. A population reservation is written before labels are read to
`data/experiments/reporting-consumption.sqlite3` in the repository root,
independent of `--output`, run IDs, and freeze filenames. Event reservations
also reject overlapping populations before scoring. Keep this ledger with the
research workspace; copying a freeze or changing its candidate does not reset
consumption. Failed and interrupted reservations remain consumed, so errors or
inconclusive results do not authorize repeated fixed-sample tests. A new disjoint
population or a
prospectively specified sequential design requires a separate decision.

## Readiness and remaining decisions

Preparation acceptance requires verified synthetic artifacts from every track,
bounded exploratory adapter execution, and passing role, invariance, corruption,
reload, calibration, and uncertainty tests. The readiness output enumerates
role/time coverage, player concentration and deck variation, form/tower/pattern
support, complete player windows, and field-by-identity mechanics gaps.

Before development, finalize data cutoffs, source-era compatibility, quantitative
mechanics sourcing or disabled status, precision/support requirements, resource
budgets, and the comparison thresholds. Before reporting, specify candidates,
the test family/alternative, assets, and the future interval. Production
integration then needs separately reviewed training/serving extraction,
artifact compatibility, calibrated inference parity, coverage, and latency.

Verification commands:

```sh
uv run ruff check .
uv run ruff format --check .
uv run --extra ml pyright
uv run --extra ml pytest
uv run --extra ml python -m experiments --synthetic
```
