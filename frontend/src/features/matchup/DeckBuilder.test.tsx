import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it } from "vitest";

import type { MatchupDeck } from "../../api/deckMatchup";
import { matchupOptions } from "../../test/matchupFixtures";
import { DeckBuilder } from "./DeckBuilder";

function Builder({ towers = true }: { towers?: boolean }) {
  const [deck, setDeck] = useState<MatchupDeck>({ cards: [], tower: null });
  return (
    <DeckBuilder
      label="Deck A"
      cards={matchupOptions.cards}
      towers={towers ? matchupOptions.towers : []}
      deck={deck}
      onChange={setDeck}
    />
  );
}

describe("DeckBuilder", () => {
  it("searches card forms and prevents selecting the same root twice", async () => {
    const user = userEvent.setup();
    render(<Builder />);
    const search = screen.getByRole("searchbox", { name: "Deck A search cards" });
    await user.type(search, "evolution");
    expect(screen.queryByRole("button", { name: "Add Knight to Deck A" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Add Evo Knight to Deck A" }));

    expect(search).toHaveValue("");
    expect(screen.getByText("1 / 8")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Add Knight to Deck A" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Add Hero Knight to Deck A" })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "Remove Evo Knight from Deck A" }));
    expect(screen.getByRole("button", { name: "Add Knight to Deck A" })).toBeEnabled();
    expect(screen.getByText("0 / 8")).toBeInTheDocument();
  });

  it("caps a deck at eight cards and permits removal", async () => {
    const user = userEvent.setup();
    render(<Builder />);
    for (const card of matchupOptions.cards.slice(0, 8)) {
      await user.click(screen.getByRole("button", { name: `Add ${card.name} to Deck A` }));
    }
    expect(screen.getByText("8 / 8")).toBeInTheDocument();
    expect(screen.getByRole("searchbox")).toBeDisabled();
    expect(screen.queryByRole("group", { name: "Deck A card choices" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Remove Knight from Deck A" }));
    expect(screen.getByRole("searchbox")).toBeEnabled();
    expect(screen.getByRole("group", { name: "Deck A card choices" })).toBeVisible();
  });

  it("keeps towers explicit and offers an empty search state", async () => {
    const user = userEvent.setup();
    render(<Builder />);
    const tower = screen.getByRole("combobox", { name: "Deck A tower troop" });
    expect(tower).toHaveValue("");
    await user.selectOptions(tower, "cannoneer:tower");
    expect(tower).toHaveValue("cannoneer:tower");
    await user.type(screen.getByRole("searchbox"), "nonexistent");
    expect(screen.getByRole("status")).toHaveTextContent("No matching cards.");
  });

  it("omits tower controls for deck-only models", () => {
    render(<Builder towers={false} />);
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
  });
});
