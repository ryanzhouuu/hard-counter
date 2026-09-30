# Model input contracts

Attention supports eight-card `attention-card-schema:v1` and nine-token
`attention-card-schema:v2` inputs. Catalog and attribute versions describe the
selected data; vocabulary sizes derive from that catalog. Legacy training keeps
the packaged June defaults. Official snapshot preparation uses current inputs.

## Catalog snapshots

The packaged `infrastructure/clash_royale/cards.json` snapshot includes the 182
deck identities released through September 26, 2026: the June identities plus
Ronin, Minion Giant, Elite Barbarians Evolution, and Hero Valkyrie, Berserker,
and Ice Wizard. The Kaggle catalog retains its original 176 identities.
Its companion `clash_royale/attributes.json` adds Ronin and Minion Giant and sets
Void to five elixir for current-era training. June inputs retain Void at three.
Spirit Empress retains the existing three-elixir feature convention for its
three/six-elixir deployment mechanic. Forms continue sharing base attributes.

`clash_royale/towers.json` separately catalogs Tower Princess, Cannoneer,
Dagger Duchess, and Royal Chef with their official API IDs and rarity-relative
level offsets. They have no deploy elixir cost and occupy a separate tower slot.

Catalog JSON contains `catalog_version` and `entries`. Each entry is
`[source_id, source_name, card_id, form]`. Source IDs must be ordered from zero;
names and card/form identities must be unique. Evolution and hero entries need
a corresponding base or champion entry. Supported forms remain `base`,
`champion`, `evolution`, and `hero`.

Source IDs are lookup positions in this mapping, not model tokens. Model tokens
index the artifact's sorted card/form vocabulary. Adding identities may change
token numbers, so each artifact keeps its exact catalog, attributes, vocabulary,
feature definitions, and weights together.

## Schema validation

`attention-card-schema:v2` adds a typed ninth tower token after the eight sorted
deployable card tokens. Its frozen tower catalog participates in the encoding
hash. Tower attributes are zero (no deploy elixir or deck roles); the distinct
tower form and identity embeddings carry their meaning. Attention and explicit
own/opposing pair terms include towers, allowing learned card/tower interactions.
V1 inputs and artifacts retain their eight-token layout and original hash.

The schema checks catalog hashes and versions, complete base-card attributes,
vocabulary alignment, token indices, and attribute vectors. It supports up to
65,536 card/form identities. Roles, elixir scaling, and form inheritance retain
their existing meanings. Models estimate matchups under an equal-skill
assumption; additional vocabulary entries alone do not establish trained support.

Training eligibility remains level 16. The current Kaggle prepared-data contract
also requires its ranked mode and winner-first rows. Supporting another catalog
does not change those source requirements.

## Attention caches

`attention-input-cache:v1` stores eight tokens per side; v2 stores eight cards
followed by one tower. Both store tokens as `uint16` and binary labels as
`uint8`. Older `uint8` token caches must be regenerated. Training rejects them
with rebuild guidance and leaves their files intact. Select a fresh
`--cache-directory` when rebuilding. Changing cache token storage does not change
an existing model artifact's vocabulary or weights. Readers verify the layout
version, vocabulary bounds, and tower slot types. In v2, support counts describe
deck-plus-tower lineups so identical decks with different towers remain distinct.

## Selecting training inputs

`model train-attention` accepts optional `--catalog` and `--attributes` JSON paths.
Without them it uses the packaged June inputs. The balance era comes from the
resolved protocol and must match the prepared dataset. Its `encoding_sha256` must
match the selected card inputs and network configuration.

```bash
uv run --extra ml clash-sos model train-attention \
  --protocol data/config/attention-temporal.json \
  --catalog data/config/card-catalog.json \
  --attributes data/config/card-attributes.json \
  --cache-directory data/cache/attention-uint16
```

The Kaggle source validator requires coverage of every identity in its frozen
June mapping, even when the model catalog contains additional identities. It
continues verifying manifest/file hashes, the era, and resolved row selections.
Official snapshots use the normalized preparation command below.

## Preparing tower-aware training inputs

The local collector reads a fixed file of player tags, one `#TAG` per line,
under ignored `data/`. Start with a selected cohort of several hundred max-level
ranked players; this is a sampling choice, not a representative-population
guarantee. The commands accept 1–800 distinct tags. Supply the official API
token through `CLASH_ROYALE_API_TOKEN`. One request is in flight at a time.
The operator-started `collect sweep` attempts each due tag once and exits. For
the selected 800-tag cohort, run it about four times per day while the local
machine is available. Gaps in highly active players' bounded logs are possible;
battles that have already left a log cannot be recovered.

```bash
uv run clash-sos collect sweep
uv run clash-sos collect report
uv run clash-sos collect export \
  --destination data/tmp/official-ranked16-pilot.jsonl \
  --dataset-version official-ranked16-pilot --balance-era 2026-09 \
  --start 2026-09-07T00:00:00Z --end 2026-09-27T00:00:00Z
```

`collect sweep` defaults to `data/collector/players.txt`, a two-second request
spacing, a one-hour minimum gap since each tag's last successful poll, and a
45-minute runtime limit. An interrupted or timed-out sweep keeps completed
polls, and the next invocation attempts unfinished tags once they are due.
Failures retain their backoff and credential rejection stops the command.
The output reports only that invocation's attempts, new matches, duplicates,
rejections, and possible gaps. `collect run` remains available for bounded
continuous polling when repeated cycles are needed for diagnosis. Adjust
timing from observed overlap and API errors. The SQLite database at
`data/collector/official.sqlite` retains normalized match variants, poll state,
and conflicts. `collect report` shows status, rejection, failure, and possible
gap counts. Exports include only conflict-free, eligible current-mode matches;
older-mode eligible matches are counted as skipped. The JSONL file is a
temporary bridge to preparation. Give each export a new destination; publishing
a Parquet snapshot leaves SQLite untouched.

`dataset prepare-official-attention` accepts normalized JSONL, one
`TowerBattleRow` per line. It does not accept raw API battle-log JSON. Each row
uses the canonical winner-first battle fields, `source_id="official-api"`, the
selected dataset version and era, eight distinct supported card identities per
side, and normalized card levels of 16. Both `side_a_tower` and `side_b_tower`
must be supported `:tower` identities with their corresponding
`side_a_tower_level` and `side_b_tower_level` equal to 16. Record towers from
the battle's `supportCards`; never substitute the current player profile or
assume Tower Princess. API rarity-relative levels must be normalized first.
Current official Path of Legends logs use `Ranked1v1_NewArena2`; the default
`official-ranked16-schema:v2` preserves that value. The older
`official-ranked16-schema:v1` remains available through `--official-schema-version`
for snapshots containing `Ranked1v1_NewArena`.

`event_key` and `fingerprint` identify the source battle. Repeated fingerprints
or event keys fail preparation, including conflicting observations. The retained
`archive_member` and zero-based `row_number` locate normalized input rows; they
do not claim historical collection provenance. The snapshot inventory records
file hashes, counts, versions, era, and temporal bounds.

Preparation validates every row in bounded batches and publishes an immutable
snapshot with `canonical.parquet`, `splits-temporal.parquet`, `manifest.json`,
`protocol.json`, and `feature-schema.json`. All three partitions must be
nonempty; training needs at least two distinct timestamps for fit/watch
selection. Bounds are timezone-aware and half-open. No rows outside the declared
interval are silently dropped. Kaggle rows lack towers and cannot train v2.

```bash
uv run clash-sos dataset prepare-official-attention \
  --source data/tmp/official-ranked16-pilot.jsonl \
  --destination data/processed/official-ranked16-v1 \
  --dataset-version official-ranked16-v1 --balance-era 2026-09 \
  --start 2026-09-07T00:00:00Z --train-end 2026-09-21T00:00:00Z \
  --validation-end 2026-09-24T00:00:00Z --end 2026-09-27T00:00:00Z

uv run --extra ml clash-sos model train-attention \
  --dataset data/processed/official-ranked16-v1 \
  --protocol data/processed/official-ranked16-v1/protocol.json \
  --feature-schema data/processed/official-ranked16-v1/feature-schema.json \
  --cache-directory data/cache/official-ranked16-v1 \
  --model-version official-ranked16-attention-v1
```

The dates above illustrate a single-era snapshot; select bounds for the actual
collected population. `--network-config` on preparation selects an architecture
before resolving the protocol. Training's `--feature-schema` freezes the entire
input and network contract and cannot be combined with `--catalog`,
`--attributes`, or `--network-config`. The selected schema must match the
protocol's era and encoding hash. Publishing a candidate does not promote it
to live serving; select its artifact through the existing model setting after
evaluation. Current catalog coverage alone does not provide learned estimates.

## Live mapping and model coverage

Set optional `CLASH_SOS_LIVE_CATALOG_PATH` to a catalog JSON file to select the
live name/form mapping. It defaults to the packaged current mapping. Actual new
API identities and representations still require validation before adding them.

Parsing uses this mapping; scoring uses the selected artifact's frozen
vocabulary. Battles containing a mapped identity absent from that vocabulary
remain visible with `model_coverage` as their skip reason. They do not contribute
to rolling metrics or prevent supported battles from being scored. Unknown
identities/forms remain excluded without substituting base cards. Live mode and
level handling retain their existing behavior.
Live parsing separately preserves each side's `supportCards` identity and
normalized tower level. Missing, malformed, or unknown towers are never assumed
to be Tower Princess. They do not change legacy deck-only eligibility.
For v2 artifacts, missing/unknown towers exclude the battle with a visible tower
reason. Reports include both towers and `model.input_scope` (`deck_only` or
`deck_and_tower`). Live estimates still extrapolate across levels and modes;
training requires normalized level 16. Direct attention predictions accept
`side_a_tower` and `side_b_tower` identity strings and require both for v2.
