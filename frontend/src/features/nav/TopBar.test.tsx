import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { saveRecentPlayer } from "../../lib/recentPlayers";
import { TopBar } from "./TopBar";

describe("TopBar", () => {
  it("marks the current page and hides the tag box without a handler", () => {
    render(<TopBar page="matchup" />);
    expect(screen.getByRole("link", { name: "Matchup" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("link", { name: "Players" })).not.toHaveAttribute("aria-current");
    expect(screen.getByRole("link", { name: "Players" })).toHaveAttribute("href", "./");
    expect(screen.queryByRole("textbox", { name: "Player tag" })).not.toBeInTheDocument();
  });

  it("returns Players to the most recent report", () => {
    saveRecentPlayer({ tag: "ABC123", name: "Ryan", window: 25 });
    render(<TopBar page="players" tag="ABC123" onLookUp={vi.fn()} />);
    expect(screen.getByRole("link", { name: "Players" })).toHaveAttribute(
      "href",
      "?tag=ABC123&window=25",
    );
    expect(screen.getByRole("textbox", { name: "Player tag" })).toHaveValue("ABC123");
  });
});
