# Clash SoS

Clash SoS is a project for understanding how difficult a player's Clash Royale
matches were. The goal is a tool that combines deck matchup estimates with match
history to show rolling strength of schedule, expected wins, and performance
relative to expectation.

## Current capabilities

- Audit and prepare ranked match data from the Kaggle version 6 archive.
- Train and evaluate deck matchup models, including LightGBM, card-pair baselines, and an optional attention model.
- Look up a player on demand through the backend's `/api/player-analysis` route.
  The web app shows the resulting report, including each battle's decks and
  estimated win chance.

Model outputs are estimates of deck matchups, not measurements of player skill.
The attention model was trained on June 2026 ranked level-16 battles and assumes
equal player skill. Current battle estimates extend beyond that training population;
new cards and forms are skipped. Live battle history is not saved locally.

## Development

See [SETUP.md](docs/SETUP.md) for installation, local commands, data
preparation, model training, and quality checks.
