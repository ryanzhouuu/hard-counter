import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { jsonResponse } from "../../test/fixtures";
import { matchupOptions, matchupResult } from "../../test/matchupFixtures";
import { MatchupPage } from "./MatchupPage";

afterEach(() => vi.unstubAllGlobals());

describe("MatchupPage", () => {
  it("builds two decks, predicts explicit forms and towers, and clears results on edit", async () => {
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(matchupOptions))
      .mockResolvedValue(jsonResponse(matchupResult));
    vi.stubGlobal("fetch", fetch);
    const user = userEvent.setup();
    render(<MatchupPage />);

    const predict = await screen.findByRole("button", { name: "Predict matchup" });
    expect(predict).toBeDisabled();
    for (const label of ["Deck A", "Deck B"]) {
      const names = matchupOptions.cards.slice(0, 8).map((card) => card.name);
      if (label === "Deck A") names[0] = "Evo Knight";
      for (const name of names)
        await user.click(screen.getByRole("button", { name: `Add ${name} to ${label}` }));
    }
    expect(predict).toBeDisabled();
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Deck A tower troop" }),
      "tower-princess:tower",
    );
    expect(predict).toBeDisabled();
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Deck B tower troop" }),
      "cannoneer:tower",
    );
    expect(predict).toBeEnabled();
    await user.click(predict);

    expect(await screen.findByRole("heading", { name: "Deck A is favored" })).toBeInTheDocument();
    expect(screen.getByText("65%")).toBeInTheDocument();
    expect(screen.getByText("35%")).toBeInTheDocument();
    expect(
      screen.getByText(/Estimated win chances assuming equal player skill/),
    ).toBeInTheDocument();
    expect(screen.getByText(/Model: tower-test/)).toHaveTextContent("Training era: 2026-09");
    const payload = JSON.parse(fetch.mock.calls[1][1].body);
    expect(payload.side_a.cards[0]).toBe("knight:evolution");
    expect(payload.side_b.tower).toBe("cannoneer:tower");
    await user.click(screen.getByRole("button", { name: "Remove Evo Knight from Deck A" }));
    expect(screen.queryByRole("region", { name: "Matchup prediction" })).not.toBeInTheDocument();
    expect(predict).toBeDisabled();
  });

  it("explains model failures and retries loading its choices", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValueOnce(
          jsonResponse({ detail: "The selected model artifact is unavailable." }, 503),
        )
        .mockResolvedValueOnce(jsonResponse(matchupOptions)),
    );
    const user = userEvent.setup();
    render(<MatchupPage />);
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "selected model artifact is unavailable",
    );
    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(
      await screen.findByRole("searchbox", { name: "Deck A search cards" }),
    ).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("shows catalog loading and aborts it when leaving the page", async () => {
    const fetch = vi.fn(() => new Promise<Response>(() => {}));
    vi.stubGlobal("fetch", fetch);
    const { unmount } = render(<MatchupPage />);
    expect(screen.getByRole("status")).toHaveTextContent("Loading cards and model");
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1));
    unmount();
    const calls = vi.mocked(globalThis.fetch).mock.calls;
    expect(calls[0][1]?.signal?.aborted).toBe(true);
  });
});
