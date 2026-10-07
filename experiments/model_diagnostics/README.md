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
