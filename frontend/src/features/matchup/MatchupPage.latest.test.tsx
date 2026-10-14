import { act, fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { saveRecentPlayer } from "../../lib/recentPlayers";
import { jsonResponse, makeAnalysis, makeBattle } from "../../test/fixtures";
import { matchupOptions } from "../../test/matchupFixtures";
import { MatchupPage } from "./MatchupPage";

const report = makeAnalysis({
  battles: [
    makeBattle({
      player_cards: matchupOptions.cards.slice(0, 8),
      player_tower: { ...matchupOptions.towers[0], level: 11 },
    }),
  ],
});

afterEach(() => vi.unstubAllGlobals());

async function setup() {
  saveRecentPlayer({ tag: "ABC123", name: "Ryan", window: 10 });
  const replies: ((response: Response) => void)[] = [];
  const fetch = vi
    .fn()
    .mockResolvedValueOnce(jsonResponse(matchupOptions))
    .mockImplementation(() => new Promise<Response>((resolve) => replies.push(resolve)));
  vi.stubGlobal("fetch", fetch);
  const user = userEvent.setup();
  const page = render(<MatchupPage />);
  await screen.findAllByRole("button", { name: "Use my latest" });
  const deckA = within(screen.getByRole("group", { name: /^Deck A \d/ }));
  const deckB = within(screen.getByRole("group", { name: /^Deck B \d/ }));
  async function finish(index = 0, response = jsonResponse(report)) {
    await act(async () => replies[index](response));
  }
  return { ...page, user, fetch, deckA, deckB, finish };
}

describe("latest deck loading", () => {
  it("guards duplicate clicks and keeps each deck's loading state separate", async () => {
    const { fetch, deckA, deckB, finish } = await setup();
    const button = deckA.getByRole("button", { name: "Use my latest" });
    act(() => {
      fireEvent.click(button);
      fireEvent.click(button);
    });
    expect(fetch).toHaveBeenCalledTimes(2);
    expect(deckA.getByRole("button", { name: "Loading latest…" })).toBeDisabled();
    expect(deckB.getByRole("button", { name: "Use my latest" })).toBeEnabled();
    await finish();
    expect(deckA.getByText(/^8 \/ 8/)).toBeInTheDocument();
    expect(deckA.getByRole("button", { name: "Use my latest" })).toBeEnabled();
  });

  it.each(["add", "remove", "clear", "tower", "swap"])(
    "preserves a %s edit made while the latest deck is loading",
    async (edit) => {
      const { user, fetch, deckA, deckB, finish } = await setup();
      await user.click(deckA.getByRole("button", { name: "Add Knight to Deck A" }));
      await user.click(deckA.getByRole("button", { name: "Use my latest" }));
      if (edit === "add")
        await user.click(deckA.getByRole("button", { name: "Add Archers to Deck A" }));
      if (edit === "remove")
        await user.click(deckA.getByRole("button", { name: "Remove Knight from Deck A" }));
      if (edit === "clear") await user.click(deckA.getByRole("button", { name: "Clear Deck A" }));
      if (edit === "tower")
        await user.selectOptions(deckA.getByRole("combobox"), "cannoneer:tower");
      if (edit === "swap") await user.click(screen.getByRole("button", { name: "Swap decks" }));

      const expectedCount = edit === "add" ? 2 : edit === "tower" ? 1 : 0;
      await finish();
      expect(deckA.getByText(new RegExp(`^${expectedCount} / 8`))).toBeInTheDocument();
      expect(deckA.getByRole("combobox")).toHaveValue(edit === "tower" ? "cannoneer:tower" : "");
      expect(deckB.getByText(new RegExp(`^${edit === "swap" ? 1 : 0} / 8`))).toBeInTheDocument();
      expect(fetch.mock.calls[1][1].signal.aborted).toBe(true);
    },
  );

  it("ignores a cancelled failure while a newer request is still loading", async () => {
    const { user, deckA, finish } = await setup();
    await user.click(deckA.getByRole("button", { name: "Use my latest" }));
    await user.click(deckA.getByRole("button", { name: "Add Knight to Deck A" }));
    await user.click(deckA.getByRole("button", { name: "Use my latest" }));
    await finish(0, jsonResponse({ detail: "Old request failed" }, 502));
    expect(screen.queryByText("Old request failed")).not.toBeInTheDocument();
    expect(deckA.getByRole("button", { name: "Loading latest…" })).toBeDisabled();
    await finish(1);
    expect(deckA.getByText(/^8 \/ 8/)).toBeInTheDocument();
  });

  it("does not cancel the other deck when one side is edited", async () => {
    const { user, fetch, deckA, deckB, finish } = await setup();
    await user.click(deckA.getByRole("button", { name: "Use my latest" }));
    await user.click(deckB.getByRole("button", { name: "Use my latest" }));
    await user.click(deckA.getByRole("button", { name: "Add Knight to Deck A" }));
    await finish(0);
    await finish(1);
    expect(deckA.getByText(/^1 \/ 8/)).toBeInTheDocument();
    expect(deckB.getByText(/^8 \/ 8/)).toBeInTheDocument();
    expect(fetch.mock.calls[1][1].signal.aborted).toBe(true);
    expect(fetch.mock.calls[2][1].signal.aborted).toBe(false);
  });

  it("aborts pending latest-deck requests when leaving the page", async () => {
    const { user, fetch, deckA, deckB, unmount } = await setup();
    await user.click(deckA.getByRole("button", { name: "Use my latest" }));
    await user.click(deckB.getByRole("button", { name: "Use my latest" }));
    unmount();
    expect(fetch.mock.calls[1][1]?.signal?.aborted).toBe(true);
    expect(fetch.mock.calls[2][1]?.signal?.aborted).toBe(true);
  });
});
