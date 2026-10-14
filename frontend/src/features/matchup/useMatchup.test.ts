import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { MatchupDeck } from "../../api/deckMatchup";
import { jsonResponse } from "../../test/fixtures";
import { matchupOptions, matchupResult } from "../../test/matchupFixtures";
import { useMatchup } from "./useMatchup";

const complete: MatchupDeck = {
  cards: matchupOptions.cards.slice(0, 8).map((card) => card.identity),
  tower: "tower-princess:tower",
};

afterEach(() => vi.unstubAllGlobals());

describe("useMatchup", () => {
  it("swaps the two decks and drops the old result", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(matchupOptions)));
    const { result } = renderHook(useMatchup);
    await waitFor(() => expect(result.current.options).not.toBeNull());
    const other: MatchupDeck = { cards: complete.cards.slice(0, 3), tower: null };
    act(() => {
      result.current.changeDeck("side_a", complete);
      result.current.changeDeck("side_b", other);
    });
    act(() => result.current.swap());
    expect(result.current.decks).toEqual({ side_a: other, side_b: complete });
    expect(result.current.result).toBeNull();
  });

  it("requires complete decks and towers before posting a prediction", async () => {
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(matchupOptions))
      .mockResolvedValue(jsonResponse(matchupResult));
    vi.stubGlobal("fetch", fetch);
    const { result } = renderHook(useMatchup);
    await waitFor(() => expect(result.current.options).not.toBeNull());
    act(() => {
      result.current.changeDeck("side_a", complete);
      result.current.changeDeck("side_b", { ...complete, tower: null });
    });
    expect(result.current.ready).toBe(false);
    await act(() => result.current.predict());
    expect(fetch).toHaveBeenCalledTimes(1);
    act(() => result.current.changeDeck("side_b", complete));
    expect(result.current.ready).toBe(true);
    await act(() => result.current.predict());
    expect(result.current.result).toEqual(matchupResult);
    act(() => result.current.changeDeck("side_a", { ...complete, tower: "cannoneer:tower" }));
    expect(result.current.result).toBeNull();
  });

  it("ignores a prediction that completes after a deck edit", async () => {
    let settle!: (response: Response) => void;
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(matchupOptions))
      .mockImplementation(
        () =>
          new Promise<Response>((resolve) => {
            settle = resolve;
          }),
      );
    vi.stubGlobal("fetch", fetch);
    const { result } = renderHook(useMatchup);
    await waitFor(() => expect(result.current.options).not.toBeNull());
    act(() => {
      result.current.changeDeck("side_a", complete);
      result.current.changeDeck("side_b", complete);
    });
    let prediction!: Promise<void>;
    act(() => {
      prediction = result.current.predict();
    });
    expect(result.current.predicting).toBe(true);
    act(() => result.current.changeDeck("side_a", { cards: [], tower: null }));
    expect(result.current.predicting).toBe(false);
    expect(fetch.mock.calls[1][1].signal.aborted).toBe(true);
    await act(async () => {
      settle(jsonResponse(matchupResult));
      await prediction;
    });
    expect(result.current.result).toBeNull();
  });

  it("retries failed catalog loading and allows deck-only models without towers", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValueOnce(jsonResponse({ detail: "Model unavailable" }, 503))
        .mockResolvedValueOnce(
          jsonResponse({
            ...matchupOptions,
            model: { ...matchupOptions.model, input_scope: "deck_only" },
            towers: [],
          }),
        ),
    );
    const { result } = renderHook(useMatchup);
    await waitFor(() => expect(result.current.optionsError?.message).toBe("Model unavailable"));
    act(() => result.current.retryOptions());
    await waitFor(() => expect(result.current.options).not.toBeNull());
    expect(result.current.optionsError).toBeNull();
    act(() => {
      result.current.changeDeck("side_a", { ...complete, tower: null });
      result.current.changeDeck("side_b", { ...complete, tower: null });
    });
    expect(result.current.ready).toBe(true);
  });

  it("retains decks after a prediction failure and supports retry", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValueOnce(jsonResponse(matchupOptions))
        .mockResolvedValueOnce(jsonResponse({ detail: "No coverage" }, 422))
        .mockResolvedValueOnce(jsonResponse(matchupResult)),
    );
    const { result } = renderHook(useMatchup);
    await waitFor(() => expect(result.current.options).not.toBeNull());
    act(() => {
      result.current.changeDeck("side_a", complete);
      result.current.changeDeck("side_b", complete);
    });
    await act(() => result.current.predict());
    expect(result.current.error?.message).toBe("No coverage");
    expect(result.current.decks.side_a).toEqual(complete);
    await act(() => result.current.predict());
    expect(result.current.error).toBeNull();
    expect(result.current.result).toEqual(matchupResult);
  });
});
