import { afterEach, describe, expect, it, vi } from "vitest";

import { jsonResponse } from "../test/fixtures";
import { matchupOptions, matchupResult } from "../test/matchupFixtures";
import { fetchMatchupOptions, predictMatchup } from "./deckMatchup";

afterEach(() => vi.unstubAllGlobals());

describe("matchup API", () => {
  it("loads the selected model choices with an abort signal", async () => {
    const fetch = vi.fn().mockResolvedValue(jsonResponse(matchupOptions));
    vi.stubGlobal("fetch", fetch);
    const { signal } = new AbortController();

    expect(await fetchMatchupOptions(signal)).toEqual(matchupOptions);
    expect(fetch).toHaveBeenCalledWith("/api/matchup-options", { signal });
  });

  it("posts identities and explicit towers without changing orientation", async () => {
    const fetch = vi.fn().mockResolvedValue(jsonResponse(matchupResult));
    vi.stubGlobal("fetch", fetch);
    const body = {
      side_a: { cards: ["knight:evolution"], tower: "tower-princess:tower" },
      side_b: { cards: ["archers:base"], tower: "cannoneer:tower" },
    };

    expect(await predictMatchup(body)).toEqual(matchupResult);
    expect(fetch).toHaveBeenCalledWith("/api/matchup-prediction", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: undefined,
    });
  });

  it("retains backend error details", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse({ detail: "Model unavailable" }, 503)),
    );
    await expect(fetchMatchupOptions()).rejects.toMatchObject({
      message: "Model unavailable",
      status: 503,
    });
  });

  it("explains validation failures and unreachable servers", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ detail: [] }, 422)));
    await expect(fetchMatchupOptions()).rejects.toMatchObject({
      message: "Check both decks and try again.",
      status: 422,
    });
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    await expect(fetchMatchupOptions()).rejects.toMatchObject({
      message: "Could not reach the server. Try again.",
      status: 0,
    });
  });

  it("preserves cancellation", async () => {
    const controller = new AbortController();
    controller.abort();
    const error = new DOMException("Aborted", "AbortError");
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(error));
    await expect(fetchMatchupOptions(controller.signal)).rejects.toBe(error);
  });
});
