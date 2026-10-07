# Capacity and player diagnostics

These exploratory models investigate overfitting and removable player effects.
They do not change existing study contracts or produce promotion candidates.

`ModelSpec` selects main card/tower effects or the existing explicit pair model,
with no player control, past-only smoothed win-rate control, or joint player
intercepts. At inference, `matchup_logits` omits the player term. This is an
additive equal-skill assumption, not proof that the components are identified.

`card_l2`, `pair_l2`, and `player_l2` independently penalize their weights using
`lambda * sum(weight squared) / 2`, added to mean binary cross entropy.
Joint effects are centered within training opponent components before the
penalty. These penalties are not AdamW weight decay and are not divided by the
number of parameters. The history penalty applies to its single coefficient.
The main-effects model freezes the zero pair tables. The default spec preserves
the existing explicit model and zero objective penalty for reproduction.

Use `recipe` with `fit_research`, `penalty=0`, and `scale_features=False` so
player IDs and past-rating differences retain their original interpretation.
All preprocessing and player vocabularies restart on refit rows only.

`scoring.evaluate` fits temperatures only on calibration rows and publishes raw,
full-score calibrated, shared-temperature matchup, and separately calibrated
matchup predictions. Failed calibration retains its raw evidence but supplies
no calibrated output or eligible full-score selection loss. Full-score loss is
a diagnostic selection target, not evidence of equal-skill identification.
Reports include UTC days, seen-player counts, and descriptive history support.

Trial artifacts bind the model and optimizer settings, seed, optional shuffle
seed, frozen population, encoding, protocol, source revision, code digest, and
lockfile hash. Each output is inventoried. Publication is atomic and exclusive;
resume rejects changed inputs or output corruption. Failed/time-limited trials
remain inspectable and cannot be silently overwritten or promoted.

`trial.run_trial` uses the existing watch-selection/restart-refit loop and checks
checkpoint reload and side-swap parity before completion. `load_model` requires
the same verified session and source code. Negative controls permute labels
separately within selection-fit and watch populations, reuse those labels for
refit, and preserve real calibration/development outcomes. The original source
inventory stays intact; the shuffle seed and transformed refit digest are saved.

Recovery tests use sampled binary outcomes, correlated deck assignments, and
many one-game players. They check the known matchup logit before and after
full-score calibration. A separate construction demonstrates two different
matchup/player decompositions with identical observed logits when players never
switch decks. Passing recovery in the supported simulation does not resolve
confounding in an observational dataset.

The staged schedule contains thirteen baseline settings: an exact optimizer
reference (20 epochs, patience 3), four main-effects settings, and eight pair
settings. Main models use card L2 0.01/0.1; pair models use card L2 0.01 and pair
L2 0.001/0.01/0.1/1. Both search learning rates 0.001/0.003. Other optimizer
settings are bound in the batch definition (defaults: 80 epochs, patience 8).
The retained backbone gets history/joint controls at player L2 0.001/0.01/0.1.
One training-label shuffle and two extra seeds for each of three finalists bring
the cap to 26 fits. Full calibrated outcome loss selects diagnostic settings;
unavailable calibrations are excluded, and fixed grid order breaks exact ties.

Batch execution freezes the grid and decisions, checkpoints its cumulative runtime,
verifies completed trials on resume, and refuses concurrent publishers.

Run the bounded batch with explicit frozen search inputs:

```sh
uv run --extra ml python -m experiments.model_diagnostics \
  --dataset data/processed/your-snapshot \
  --protocol data/experiments/inputs/search-protocol.json \
  --schema data/experiments/inputs/explicit-schema.json \
  --cache data/experiments/inputs/cache \
  --mechanics experiments/mechanics/inputs/2026-10-05-partial-r9.json \
  --row-cap 40000 --output data/experiments/model-diagnostics/your-run
```

Add `--check` to verify inputs and role counts without training. Search input
loading excludes physical test/reporting rows. CPU training uses two Torch
threads. Models, predictions, reports, and manifests live under `trials/` in the
output directory. The batch definition freezes the grid, selection rule, input
identity, and eight-hour budget before fitting. `progress.json` checkpoints
accumulated runtime; a resume verifies completed trial inventories and decisions,
and does not reset the budget. A new trial starts only when its full fit time
limit fits within the remaining budget. Serialization/verification may add some
overhead beyond the fit time limit. Concurrent publishers are refused.

A completed batch writes `summary.md` and `summary.json` with seed-specific paired
comparisons, constant-50% scores, learning curves, UTC day and history slices,
and both skill-removed calibration conventions. Seeds are reported separately;
they are not independent battle samples or a fitted ensemble. Intervals are
exploratory, conditional on the trained models, and omit day/meta dependence.
These assets cannot be passed to the existing candidate promotion flow.
