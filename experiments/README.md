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
- Add experiment tests to the normal quality checks when executable code is
  introduced. Tests must use synthetic data and run without local datasets or
  API credentials.
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
