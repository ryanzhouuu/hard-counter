# Clash SoS

Clash Royale matchup analysis for rolling strength of schedule, expected wins, and performance above expectation.

The project is in its initial scaffold phase. The backend, web application, CLI, workers, and notebooks will share one versioned analytics model.

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

## Planned interfaces

- FastAPI HTTP API
- React and TypeScript dashboard
- Typer CLI
- Jupyter notebooks
