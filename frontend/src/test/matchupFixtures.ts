import type { MatchupOptions, MatchupResult } from "../api/deckMatchup";

const matchupOptions: MatchupOptions = {
  model: {
    model_version: "tower-test",
    dataset_version: "test-data",
    catalog_version: "test-cards",
    training_era_id: "2026-09",
    input_scope: "deck_and_tower",
  },
  cards: [
    ...["Knight", "Archers", "Goblins", "Giant", "Minions", "Balloon", "Witch", "Bomber"].map(
      (name) => ({
        identity: `${name.toLowerCase()}:base`,
        name,
        form: "base",
        icon_url: null,
        elixir: 3,
      }),
    ),
    {
      identity: "knight:evolution",
      name: "Evo Knight",
      form: "evolution",
      icon_url: null,
      elixir: 3,
    },
    { identity: "knight:hero", name: "Hero Knight", form: "hero", icon_url: null, elixir: 3 },
    {
      identity: "archers:evolution",
      name: "Evo Archers",
      form: "evolution",
      icon_url: null,
      elixir: 3,
    },
    {
      identity: "giant:evolution",
      name: "Evo Giant",
      form: "evolution",
      icon_url: null,
      elixir: 5,
    },
    {
      identity: "archer-queen:champion",
      name: "Archer Queen",
      form: "champion",
      icon_url: null,
      elixir: 5,
    },
    {
      identity: "golden-knight:champion",
      name: "Golden Knight",
      form: "champion",
      icon_url: null,
      elixir: 4,
    },
    { identity: "mirror:base", name: "Mirror", form: "base", icon_url: null, elixir: null },
  ],
  towers: [
    { identity: "tower-princess:tower", name: "Tower Princess", icon_url: null },
    { identity: "cannoneer:tower", name: "Cannoneer", icon_url: null },
  ],
};

const matchupResult: MatchupResult = {
  model: matchupOptions.model,
  side_a_win_probability: 0.65,
  side_b_win_probability: 0.35,
};

export { matchupOptions, matchupResult };
