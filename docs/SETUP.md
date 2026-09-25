# Development setup

Run the commands below from the repository root.

## Toolchain

- Python 3.12 managed by `uv`
- Node.js 24.18.1 and `pnpm` 11.20.0
- PostgreSQL 17 and Docker Compose for database-backed development

## Install dependencies

Install the listed toolchain versions, then run:

```bash
uv sync --locked --dev --extra ml
pnpm install --frozen-lockfile
cp .env.example .env
```

Live lookup does not use PostgreSQL or save battles. Start the optional database
with `docker compose up -d postgres` for database-backed development. The
example environment file contains its local URL.

If using a non-neural artifact, the smaller default Python install is enough:

```bash
uv sync --locked --dev
```

Linux installs the CPU-only PyTorch build used by CI. macOS installs the standard
build with MPS support when the host makes it available. The default dependency
set remains torch-free for the API, CLI, and existing predictors.

## Run locally

Use separate terminals for the API and web application:

```bash
uv run clash-sos-api
pnpm web:dev
```

The API listens on `http://127.0.0.1:8000`, and the Vite development server prints its local URL when it starts.

The executable shells can be checked without starting a server:

```bash
uv run clash-sos version
uv run clash-sos-worker status
```

For live API lookup, set `CLASH_ROYALE_API_TOKEN` in the ignored `.env` file.
Create the key for the public IP address of the machine running the API, and keep
it on the backend. Set `CLASH_SOS_ACTIVE_MODEL_PATH` to switch the model artifact
without changing code. It defaults to `models/kaggle-v6-ranked16-attention-v1`;
that ignored artifact must exist locally. The attention model needs the optional
`ml` dependencies shown above.

Request an on-demand report at `/api/player-analysis?tag=%23PLAYER_TAG&window=5`.
The tag can also omit `#`; `window` accepts 1–30 eligible recent battles. The
response includes recent battles, per-battle deck estimates, and rolling schedule
metrics when enough eligible battles are available. Only decisive 1v1 battles
with two complete, known eight-card decks are scored. Current battles use the
selected model beyond its June 2026 training population; treat these as
extrapolated estimates. The lookup does not store battle history.

## Prepare ranked data

Audit the pinned Kaggle version 6 archive and write its ignored local manifest:

```bash
uv run clash-sos dataset audit-kaggle-v6
```

The command verifies the archive identity, member checksums, card catalog, and all Parquet
schemas before writing `data/metadata/kaggle-v6-dataset.json`. It profiles the archive with
bounded memory without preparing match records for model training.

Print the same audit facts without writing a manifest:

```bash
uv run clash-sos dataset inspect-kaggle-v6
```

Prepare a versioned processed dataset. Temporal cutovers are required and must be
timezone-aware ISO-8601 timestamps:

```bash
uv run clash-sos dataset prepare-kaggle-v6 \
  --train-end 2026-06-21T00:00:00+00:00 \
  --validation-end 2026-06-25T00:00:00+00:00
```

The command records the effective options in the processed manifest and fails before doing
work if `data/processed/kaggle-v6-ranked16-v2` already exists.

Validate a prepared dataset in place without rewriting it:

```bash
uv run clash-sos dataset verify-kaggle-v6
```

## Train matchup models

Train a matchup model on the prepared dataset and write a versioned
artifact. By default, the command trains a LightGBM booster with card presence,
deck summaries, and a shrunk cluster-matchup rate.
The command fits temporal train only, scores temporal and player-disjoint
splits, and fails if the destination already exists:

```bash
uv run clash-sos model train
```

The default artifact path is `models/kaggle-v6-ranked16-lightgbm-v3`.
`model train-lgbm` runs that same fit. Retrain a frozen baseline with
`--promoted-model card_pair` or `--promoted-model card_log_odds` and a new
`--destination`.

### Attention matchup experiments

Install the optional `ml` dependencies before running the deck-only attention
trainer. Supply a JSON protocol file describing the processed dataset and the
training and evaluation slices. The command checks its hashes, UTC time bounds,
row counts, and row identities before fitting. If you already have a verified
attention cache, its `manifest.json` contains the protocol object that can be
saved as a separate JSON file.

```bash
uv run --extra ml clash-sos model train-attention \
  --protocol data/config/attention-temporal-v1.json \
  --cache-directory data/cache/attention-temporal-v1 \
  --destination models/kaggle-v6-ranked16-attention-v1
```

The command reuses a compatible cache or creates it from the prepared dataset.
It uses the protocol's watch rows to select the number of training epochs, then
restarts training on the combined fit and watch rows. It scores the development
rows and, when specified, the reporting rows.
Progress goes to stderr; the completed destination goes to stdout. A destination
or output workspace that already exists is refused. Use `--network-config` for a
validated network configuration JSON file when testing ablations, and provide a
protocol whose encoding hash matches that configuration. Run
`clash-sos model train-attention --help` for optimizer, device, and batch options.

The artifact includes CPU weights, the card schema and catalog, fit history,
evaluation metrics, and aligned prediction Parquet files. Its probabilities are
deck-only matchup estimates under an equal-skill assumption for the June 2026
balance era and level-16 cards; they do not establish that player skill was
removed from the observed outcomes. Attention inference requires the caller to
provide the matchup's balance-era ID, which must match the artifact. The default
`model train` selection remains LightGBM.

## Quality checks

Run the same checks used by GitHub Actions:

```bash
uv run ruff check .
uv run ruff format --check .
uv run --extra ml pyright
uv run --extra ml pytest
pnpm web:lint
pnpm web:format:check
pnpm web:typecheck
pnpm web:test
pnpm web:build
```
