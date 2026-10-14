import { describe, expect, it } from "vitest";

import { makeBattle } from "../test/fixtures";
import { arrangeBattles } from "./battleView";

const items = [
  { battle: makeBattle({ opponent_name: "A", outcome: "win", win_probability: 0.7 }) },
  { battle: makeBattle({ opponent_name: "B", outcome: "loss", win_probability: 0.8 }) },
  { battle: makeBattle({ opponent_name: "C", outcome: "win", win_probability: 0.2 }) },
];
const names = (list: typeof items) => list.map(({ battle }) => battle.opponent_name);

describe("arrangeBattles", () => {
  it("filters by outcome and upset", () => {
    expect(names(arrangeBattles(items, "wins", "recent"))).toEqual(["A", "C"]);
    expect(names(arrangeBattles(items, "losses", "recent"))).toEqual(["B"]);
    expect(names(arrangeBattles(items, "upsets", "recent"))).toEqual(["B", "C"]);
    expect(names(arrangeBattles(items, "all", "recent"))).toEqual(["A", "B", "C"]);
  });

  it("sorts toughest matchups first without mutating input", () => {
    expect(names(arrangeBattles(items, "all", "chance"))).toEqual(["C", "A", "B"]);
    expect(names(items)).toEqual(["A", "B", "C"]);
  });
});
