# Clash SoS

Clash Royale matchup analysis for rolling strength of schedule, expected wins, and performance above expectation.

R1 is complete: the canonical analytics contracts and pinned Kaggle version 6 source are
executable and verified against the full local archive. R2, the reproducible ranked-data
foundation, is next.

## Toolchain

- Python 3.12 managed by `uv`
- Node.js 24.18.1 and `pnpm` 11.20.0
- PostgreSQL 17
- Docker Compose

## Setup

Install the listed toolchain versions, then run these commands from the repository root:

```bash
uv sync --locked --dev
pnpm install --frozen-lockfile
cp .env.example .env
docker compose up -d postgres
```

The PostgreSQL service is available at `localhost:5432`. The example environment file contains the matching local database URL.

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

Audit the pinned Kaggle version 6 archive and write its ignored local manifest:

```bash
uv run clash-sos dataset audit-kaggle-v6
```

The command verifies the archive identity, member checksums, card catalog, and all Parquet
schemas before writing `data/metadata/kaggle-v6-dataset.json`. It profiles the archive with
bounded memory and does not create the R2 processed dataset.

Print the same audit facts without writing a manifest:

```bash
uv run clash-sos dataset inspect-kaggle-v6
```

Prepare a versioned processed dataset. Temporal cutovers are required and must be
timezone-aware ISO-8601 timestamps:

```bash
uv run clash-sos dataset prepare-kaggle-v6 \
  --train-end 2026-06-10T00:00:00+00:00 \
  --validation-end 2026-06-20T00:00:00+00:00
```

The command records the effective options in the processed manifest and fails before doing
work if `data/processed/kaggle-v6-ranked16-v1` already exists.

Validate a published version in place without rewriting it:

```bash
uv run clash-sos dataset verify-kaggle-v6
```

## Quality checks

Run the same checks used by GitHub Actions:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
pnpm web:lint
pnpm web:format:check
pnpm web:typecheck
pnpm web:test
pnpm web:build
```

## Project status

- R0 repository scaffold: complete.
- R1 analytics and source-data contract: complete.
- R2 reproducible ranked-data foundation: next.
