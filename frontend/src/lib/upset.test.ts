import { describe, expect, it } from "vitest";

import { makeBattle } from "../test/fixtures";
import { upsetOf } from "./upset";

describe("upsetOf", () => {
  it("flags wins below 40% and losses above 60%", () => {
    expect(upsetOf(makeBattle({ outcome: "win", win_probability: 0.39 }))).toBe("win");
    expect(upsetOf(makeBattle({ outcome: "loss", win_probability: 0.61 }))).toBe("loss");
  });

  it("ignores expected results, boundaries, and unscored battles", () => {
    expect(upsetOf(makeBattle({ outcome: "win", win_probability: 0.4 }))).toBeNull();
    expect(upsetOf(makeBattle({ outcome: "loss", win_probability: 0.6 }))).toBeNull();
    expect(upsetOf(makeBattle({ outcome: "draw", win_probability: 0.1 }))).toBeNull();
    expect(upsetOf(makeBattle({ win_probability: null, skip_reason: "draw" }))).toBeNull();
  });
});
