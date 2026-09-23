# Clash SoS

Clash SoS is a project for understanding how difficult a player's Clash Royale
matches were. The goal is a tool that combines deck matchup estimates with match
history to show rolling strength of schedule, expected wins, and performance
relative to expectation.

## Current capabilities

- Audit and prepare ranked match data from the Kaggle version 6 archive.
- Train and evaluate deck matchup models, including LightGBM, card-pair baselines, and an optional attention model.
- Run a local web interface and API. The web interface currently shows an
  overview; the API currently exposes a health endpoint. Matchup and schedule
  views are still under development.

Model outputs are estimates of deck matchups, not measurements of player skill.
The attention model's probabilities assume equal player skill and are limited to
the June 2026 balance era and level-16 cards.

## Development

See [SETUP.md](docs/SETUP.md) for installation, local commands, data
preparation, model training, and quality checks.
