# Vercel deployment

The repository deploys as one FastAPI project. Vercel serves the built React app
from its CDN and routes API requests to a Python function on the same domain.
Use the repository root as the project's Root Directory; `vercel.json` selects
the FastAPI framework and supplies the install and build commands.

Git pushes and CLI deployments use the model release pinned in
`infra/vercel/model-release.json`. The descriptor contains an immutable private
Vercel Blob URL and the archive's SHA-256 digest. Model bytes stay outside Git.
Each build downloads that release, checks its digest, rejects unsafe archive
members, and validates every file against the model manifest before packaging.
It replaces any stale staged model from a previous build.

## Configure private model storage

The `hard-counter-models` private Blob store is connected to Preview and
Production. Its archive contains the complete
`official-ranked16-june-finetuned-pair-20261013-v1` artifact.

For a new Vercel project, create and connect a private store:

```bash
pnpm dlx vercel blob create-store hard-counter-models --access private --yes \
  --environment preview --environment production
```

Set `BLOB_STORE_ID` to the returned store ID for both environments. Vercel supplies
`VERCEL_OIDC_TOKEN` to builds; the download uses this short-lived token first and
falls back to `BLOB_READ_WRITE_TOKEN` when no OIDC token is available. These
credentials belong in Vercel environment settings, never in Git or frontend code.
See [private Blob authentication](https://vercel.com/docs/vercel-blob/private-storage#accessing-without-the-sdk).

The archive is downloaded at build time and bundled in the Python function.
Live predictions do not fetch the model from Blob. Downloads and expanded
archives are limited to 64 MiB; authentication failures, checksum mismatches, and
invalid model files stop the build before frontend assets are published.

## Publish a different model

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
copy in ignored `infra/vercel/model/`. Package it and compute the release digest:

```bash
COPYFILE_DISABLE=1 tar -czf /tmp/model.tar.gz -C infra/vercel/model .
shasum -a256 /tmp/model.tar.gz
```

For local Blob operations, export the connected store's `BLOB_READ_WRITE_TOKEN`.
Local OIDC tokens use the Development environment, which this workflow does not
connect. Clear OIDC variables for the local upload:

```bash
VERCEL_OIDC_TOKEN= BLOB_STORE_ID= pnpm dlx vercel blob put /tmp/model.tar.gz \
  --access private --no-multipart --pathname models/YOUR_MODEL_VERSION/SHA256.tar.gz
```

Use the actual model version and digest in the pathname. Update the descriptor
with the returned URL and digest, then commit it with any corresponding code.
Publish each new archive to a new pathname; retain previous releases for
rollback. Changing only the locally staged model does not change a pinned build.

The CLI upload allowlist includes the release descriptor, provisioning scripts,
application source, build inputs, and optional staged model. It excludes `.env`,
local datasets, other models, and local working material. Model files are
packaged in the function, outside the public frontend assets. Without a release
descriptor, CLI builds can still use a locally staged artifact.

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

After publishing the pinned model and configuring Preview variables, push a
branch to the connected Git repository or deploy using the CLI:

```bash
pnpm dlx vercel
```

The command creates a preview deployment. Its build compiles the frontend,
downloads and verifies the pinned model, and copies `frontend/dist/` into ignored
`public/`.
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

After the preview checks pass, merge to `main` to trigger the Git production
deployment, or publish using Production variables with `pnpm dlx vercel --prod`.
Verify the resulting production URL with the same checks before sharing it.

Live attention inference reuses one verified CPU model per warm function
instance. A new instance still pays import and model-loading costs; published
artifact members must remain immutable. A changed directory or manifest reloads
the cached model.

To exercise the pinned build locally, export `BLOB_READ_WRITE_TOKEN` from the
connected store and leave `VERCEL_OIDC_TOKEN` unset:

```bash
pnpm web:build
uv run --no-sync python -m scripts.vercel build
```

Local checks do not establish hosted cold-start latency, Linux runtime
compatibility, or the account's Large Functions eligibility.

Vercel references: [FastAPI deployment](https://vercel.com/docs/frameworks/backend/fastapi),
[Python runtime](https://vercel.com/docs/functions/runtimes/python), and
[upload allowlists](https://vercel.com/docs/deployments/vercel-ignore#allowlist).
