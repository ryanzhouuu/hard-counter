import type { Battle, Card, PlayerAnalysis, Schedule } from "../api/playerAnalysis";

const PLAYER_DECK = [
  "Hog Rider",
  "Musketeer",
  "Knight",
  "Ice Spirit",
  "Skeletons",
  "Cannon",
  "Fireball",
  "The Log",
];
const OPPONENT_DECK = [
  "Golem",
  "Night Witch",
  "Baby Dragon",
  "Lumberjack",
  "Tornado",
  "Lightning",
  "Mega Minion",
  "Bats",
];

function cards(names: string[]): Card[] {
  return names.map((name) => ({
    name,
    icon_url: `https://api-assets.clashroyale.com/cards/300/${encodeURIComponent(name)}.png`,
  }));
}

function makeBattle(overrides: Partial<Battle> = {}): Battle {
  return {
    timestamp: "2026-09-25T12:00:00Z",
    mode: "Ranked1v1_NewArena2",
    opponent_name: "Rival",
    outcome: "win",
    player_cards: cards(PLAYER_DECK),
    opponent_cards: cards(OPPONENT_DECK),
    player_tower: null,
    opponent_tower: null,
    skip_reason: null,
    win_probability: 0.62,
    in_window: true,
    ...overrides,
  };
}

function makeSchedule(overrides: Partial<Schedule> = {}): Schedule {
  return {
    status: "available",
    requested_window: 10,
    eligible_count: 22,
    excluded_count: 2,
    strength_of_schedule: 0.46,
    expected_wins: 5.4,
    actual_wins: 7,
    performance_above_expectation: 1.6,
    ...overrides,
  };
}

function makeAnalysis(overrides: Partial<PlayerAnalysis> = {}): PlayerAnalysis {
  return {
    player: { tag: "#ABC123", name: "Ryan", trophies: 7412 },
    model: {
      model_version: "attention",
      dataset_version: "data",
      catalog_version: "catalog",
      training_era_id: "2026-06",
      input_scope: "deck_only",
    },
    schedule: makeSchedule(),
    battles: [makeBattle()],
    ...overrides,
  };
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

export { jsonResponse, makeAnalysis, makeBattle, makeSchedule };
