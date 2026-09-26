import { describe, expect, it } from "vitest";

import { modeLabel, percent, relativeTime, signedValue, skipLabel } from "./format";

describe("signedValue", () => {
  it.each([
    [1.6, "+1.6", "positive"],
    [-0.84, "−0.8", "negative"],
    [0.04, "0.0", "neutral"],
    [-0.04, "0.0", "neutral"],
  ])("formats %s", (value, text, tone) => {
    expect(signedValue(value)).toEqual({ text, tone });
  });
});

describe("percent", () => {
  it("rounds a probability to a whole percentage", () => {
    expect(percent(0.546)).toBe("55%");
  });
});

describe("relativeTime", () => {
  const now = new Date("2026-09-25T12:00:00Z");

  it.each([
    ["2026-09-25T11:59:30Z", "now"],
    ["2026-09-25T11:48:00Z", "12m"],
    ["2026-09-25T10:00:00Z", "2h"],
    ["2026-09-22T12:00:00Z", "3d"],
  ])("describes %s as %s", (timestamp, expected) => {
    expect(relativeTime(timestamp, now)).toBe(expected);
  });

  it("is empty without a timestamp", () => {
    expect(relativeTime(null, now)).toBe("");
  });
});

describe("labels", () => {
  it("describes skip reasons in plain words", () => {
    expect(skipLabel("unknown_card")).toBe("New card not yet supported");
    expect(skipLabel("model_coverage")).toBe("Card not supported by this model");
    expect(skipLabel("team_battle")).toBe("Team battle");
    expect(skipLabel("something_new")).toBe("Not scored");
    expect(skipLabel(null)).toBe("Not scored");
  });

  it("names only observed live modes", () => {
    expect(modeLabel("Ranked1v1_NewArena2")).toBe("Ranked");
    expect(modeLabel("Ladder")).toBe("Ladder");
    expect(modeLabel("Friendly")).toBeNull();
  });
});
