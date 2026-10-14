import { describe, expect, it } from "vitest";

import { cards, makeBattle } from "../test/fixtures";
import { facedMost, winSpread } from "./stats";

describe("winSpread", () => {
  it("sums matchup variance over scored window battles", () => {
    const battles = [
      makeBattle({ win_probability: 0.5 }),
      makeBattle({ win_probability: 0.8 }),
      makeBattle({ win_probability: 0.9, in_window: false }),
      makeBattle({ win_probability: null, skip_reason: "draw" }),
    ];
    expect(winSpread(battles)).toBeCloseTo(Math.sqrt(0.25 + 0.16));
  });

  it("is zero without scored battles", () => {
    expect(winSpread([])).toBe(0);
  });
});

describe("facedMost", () => {
  it("ranks opponent cards by frequency then name with mean win chance", () => {
    const battles = [
      makeBattle({ opponent_cards: cards(["Hog Rider", "Arrows"]), win_probability: 0.4 }),
      makeBattle({ opponent_cards: cards(["Hog Rider", "Bats"]), win_probability: 0.6 }),
      makeBattle({ opponent_cards: cards(["Hog Rider"]), in_window: false, win_probability: 0.1 }),
    ];
    expect(
      facedMost(battles, 2).map(({ card, faced, winChance }) => [card.name, faced, winChance]),
    ).toEqual([
      ["Hog Rider", 2, 0.5],
      ["Arrows", 1, 0.4],
    ]);
  });
});
