import { afterEach, describe, expect, it, vi } from "vitest";

import { makeAnalysis } from "../test/fixtures";
import {
  clearRecentPlayers,
  readRecentPlayers,
  recentFromReport,
  saveRecentPlayer,
} from "./recentPlayers";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("recent players", () => {
  it("keeps the five newest distinct tags, newest first", () => {
    for (const tag of ["A1", "B2", "C3", "D4", "E5", "F6", "B2"]) {
      saveRecentPlayer({ tag, name: tag, window: 10 });
    }
    expect(readRecentPlayers().map((player) => player.tag)).toEqual(["B2", "F6", "E5", "D4", "C3"]);
    clearRecentPlayers();
    expect(readRecentPlayers()).toEqual([]);
  });

  it("ignores malformed entries", () => {
    localStorage.setItem(
      "hard-counter:recent-players",
      JSON.stringify([
        { tag: "A1", name: "Ann", window: 10 },
        { tag: 3 },
        { tag: "B2", name: "Bo", window: 7 },
      ]),
    );
    expect(readRecentPlayers()).toEqual([{ tag: "A1", name: "Ann", window: 10 }]);
  });

  it("survives unavailable storage", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    expect(readRecentPlayers()).toEqual([]);
    expect(() => saveRecentPlayer({ tag: "A1", name: "Ann", window: 10 })).not.toThrow();
  });

  it("records only a settled report for the requested tag", () => {
    const report = makeAnalysis();
    const query = { tag: "ABC123", windowSize: 25 as const };
    expect(recentFromReport(query, report, false)).toEqual({
      tag: "ABC123",
      name: "Ryan",
      window: 25,
    });
    expect(recentFromReport(query, report, true)).toBeNull();
    expect(recentFromReport({ ...query, tag: "OTHER1" }, report, false)).toBeNull();
    expect(recentFromReport(null, report, false)).toBeNull();
  });
});
