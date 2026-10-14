import { describe, expect, it } from "vitest";

import { blockReason, slotLayout } from "./deckRules";

const base = ["a:base", "b:base", "c:base", "d:base", "e:base"];

describe("slotLayout", () => {
  it("places evolutions, then heroes, then the wild slot", () => {
    expect(slotLayout(["h:hero", ...base, "x:evolution", "y:evolution"])).toEqual([
      "x:evolution",
      "h:hero",
      "y:evolution",
      ...base,
    ]);
    expect(slotLayout(["q:champion", "r:champion", "x:evolution"]).slice(0, 3)).toEqual([
      "x:evolution",
      "q:champion",
      "r:champion",
    ]);
  });

  it("lets normal cards fill special slots once plain slots are full", () => {
    const all = [...base, "f:base", "g:base", "h:base"];
    expect(slotLayout(all)).toEqual(["f:base", "g:base", "h:base", ...base]);
    expect(slotLayout([])).toEqual(Array(8).fill(null));
  });
});

describe("blockReason", () => {
  it("explains why a card cannot be added", () => {
    expect(blockReason(["a:base"], "a:evolution")).toBe("Already in deck");
    expect(blockReason(["x:evolution", "y:evolution"], "z:evolution")).toBe("Evolution slots full");
    expect(blockReason(["h:hero", "q:champion"], "r:hero")).toBe("Hero slots full");
    expect(blockReason(["x:evolution", "y:evolution", "h:hero"], "q:champion")).toBe(
      "No special slots left",
    );
    expect(blockReason(["x:evolution", "y:evolution", "h:hero"], "f:base")).toBeNull();
    expect(blockReason([...base, "f:base", "g:base", "h:base"], "i:base")).toBe("Deck complete");
  });
});
