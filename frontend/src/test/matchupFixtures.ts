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
      (name) => ({ identity: `${name.toLowerCase()}:base`, name, form: "base" }),
    ),
    { identity: "knight:evolution", name: "Evo Knight", form: "evolution" },
    { identity: "knight:hero", name: "Hero Knight", form: "hero" },
  ],
  towers: [
    { identity: "tower-princess:tower", name: "Tower Princess" },
    { identity: "cannoneer:tower", name: "Cannoneer" },
  ],
};

const matchupResult: MatchupResult = {
  model: matchupOptions.model,
  side_a_win_probability: 0.65,
  side_b_win_probability: 0.35,
};

export { matchupOptions, matchupResult };
