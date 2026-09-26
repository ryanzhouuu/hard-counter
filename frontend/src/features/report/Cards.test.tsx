import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { makeBattle } from "../../test/fixtures";
import { CardImage } from "./CardImage";
import { DeckGrid, MiniDeck } from "./DeckGrid";

const KNIGHT = { name: "Knight", icon_url: "https://api-assets.clashroyale.com/cards/300/k.png" };

describe("CardImage", () => {
  it("shows the official card image", () => {
    render(<CardImage card={KNIGHT} />);

    expect(screen.getByRole("img", { name: "Knight" })).toHaveAttribute("src", KNIGHT.icon_url);
  });

  it("falls back to a named tile when the image fails", () => {
    render(<CardImage card={KNIGHT} />);

    fireEvent.error(screen.getByRole("img", { name: "Knight" }));

    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    expect(screen.getByText("Knight")).toBeInTheDocument();
  });

  it("uses a named tile when no image is available", () => {
    render(<CardImage card={{ name: "Future Card", icon_url: null }} />);

    expect(screen.getByText("Future Card")).toBeInTheDocument();
  });
});

describe("decks", () => {
  it("labels a full deck of named cards", () => {
    render(<DeckGrid label="You" cards={makeBattle().player_cards} isPlayer />);

    const deck = screen.getByRole("region", { name: "You deck" });
    expect(within(deck).getAllByRole("img")).toHaveLength(8);
    expect(screen.getByRole("heading", { name: "You" })).toHaveClass("deck-label-player");
  });

  it("keeps mini decks out of the accessibility tree", () => {
    render(<MiniDeck cards={makeBattle().opponent_cards} />);

    expect(screen.queryAllByRole("img")).toHaveLength(0);
  });
});
