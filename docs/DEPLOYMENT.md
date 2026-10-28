# Vercel preview deployment

The repository deploys as one FastAPI project. Vercel serves the built React app
from its CDN and routes API requests to a Python function on the same domain.
Use the repository root as the project's Root Directory; `vercel.json` selects
the FastAPI framework and supplies the install and build commands.

This initial workflow uses a CLI deployment from a machine with the selected
model artifact. The artifact stays ignored by Git. A Git-only deployment does
not contain that artifact and fails the build until a separate model-provisioning
step is added.

## Prepare the selected model

Install dependencies as described in [SETUP.md](SETUP.md), including the `ml`
extra. Set `CLASH_SOS_ACTIVE_MODEL_PATH` in your local `.env` to the model you
intend to serve, then run from the repository root:

```bash
pnpm vercel:stage
```

To choose a source explicitly:

```bash
pnpm vercel:stage models/YOUR_MODEL_VERSION
```

Staging copies and validates the complete artifact before replacing the previous
copy in ignored `infra/vercel/model/`. The CLI upload allowlist includes this
directory, the application source, and build inputs. It excludes `.env`, local
datasets, other models, and local working material. Model files are packaged in
the function, outside the public frontend assets.

## Link a Vercel project

```bash
pnpm dlx vercel login
pnpm dlx vercel link
```

Choose the intended account and project, with the repository root as its source.
The resulting `.vercel/` directory stays ignored by Git.

Configure these project variables for **Preview** and **Production** before
deploying to each environment:

| Variable                         | Value                                                                 |
| -------------------------------- | --------------------------------------------------------------------- |
| `CLASH_ROYALE_API_TOKEN`         | A backend-only key whose allowlist includes the proxy's documented IP |
| `CLASH_SOS_ROYALE_API_BASE_URL`  | `https://proxy.royaleapi.dev/v1/`                                     |
| `VERCEL_SUPPORT_LARGE_FUNCTIONS` | `1`                                                                   |

The [RoyaleAPI proxy](https://docs.royaleapi.com/proxy.html) documents
`45.79.218.79` as its allowlisted IP. It receives the API key and requests.
The existing backend base-URL setting directs player lookups and card metadata
requests through it.

PyTorch and its dependencies may exceed Vercel's standard 500 MB Python bundle
limit. [Large Functions](https://vercel.com/docs/functions/limitations#large-functions-beta)
allow up to 5 GB on Fluid compute, in public beta. They currently cannot be used
with Vercel Static IPs or Secure Compute, which is why this workflow uses the
proxy. Confirm the project's eligibility before deploying.

`vercel.json` sets the deployed model path to `infra/vercel/model` and bounds
OpenMP to one thread. It installs locked dependencies with the `ml` extra and
without the development group. Local settings and development commands continue
to use their existing model selection.

The deployment install uses `--no-editable` so the function contains the installed
backend package instead of an editable import path pointing at the build machine.

## Deploy and verify a preview

After staging the intended model and configuring Preview variables:

```bash
pnpm dlx vercel
```

The command creates a preview deployment. Its build compiles the frontend,
verifies the uploaded model, and copies `frontend/dist/` into ignored `public/`.
Vercel serves those public files from its CDN. API routes retain their `/api`
prefix; the root URL serves `index.html`, including report and matchup query
parameters.

The Vercel entrypoint mounts `public/` after the API routes so Vercel's FastAPI
collector discovers the built frontend. `[tool.vercel.fastapi.static]` excludes
collected assets from the function bundle while retaining the directory stubs
needed for startup and fallback routing.

Check the resulting preview:

1. Open `/health` to confirm the function starts. This does not verify the model
   or the upstream API key.
2. Open the Matchup page, confirm card choices load, and submit two legal decks.
   Check that the result reports the intended model version.
3. Look up a known player to verify the proxy, key, and battle analysis together.
4. Compare the first request with subsequent requests and inspect function memory
   and duration in Vercel before promoting the preview.

After the preview checks pass, publish the same staged source using Production
variables with `pnpm dlx vercel --prod`. Verify the resulting production URL with
the same checks before sharing it.

Live attention inference reuses one verified CPU model per warm function
instance. A new instance still pays import and model-loading costs; published
artifact members must remain immutable. A changed directory or manifest reloads
the cached model.

To exercise the build preparation locally after staging:

```bash
pnpm web:build
uv run --no-sync python -m scripts.vercel build
```

Local checks do not establish hosted cold-start latency, Linux runtime
compatibility, or the account's Large Functions eligibility.

Vercel references: [FastAPI deployment](https://vercel.com/docs/frameworks/backend/fastapi),
[Python runtime](https://vercel.com/docs/functions/runtimes/python), and
[upload allowlists](https://vercel.com/docs/deployments/vercel-ignore#allowlist).
