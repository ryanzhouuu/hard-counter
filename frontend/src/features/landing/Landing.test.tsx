import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { readRecentPlayers, saveRecentPlayer } from "../../lib/recentPlayers";
import { Landing } from "./Landing";
import { TagForm } from "./TagForm";

describe("Landing", () => {
  it("submits a normalized tag", async () => {
    const user = userEvent.setup();
    const onLookUp = vi.fn();
    render(<Landing onLookUp={onLookUp} />);

    expect(screen.getByRole("heading", { name: "Hard Counter" })).toBeInTheDocument();
    expect(screen.getByText(/Your tag is on your Clash Royale profile/)).toBeInTheDocument();
    await user.type(screen.getByRole("textbox", { name: "Player tag" }), "#abc123");
    await user.click(screen.getByRole("button", { name: "Look up" }));

    expect(onLookUp).toHaveBeenCalledWith("ABC123");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("explains an invalid tag without submitting", async () => {
    const user = userEvent.setup();
    const onLookUp = vi.fn();
    render(<Landing onLookUp={onLookUp} />);

    await user.type(screen.getByRole("textbox", { name: "Player tag" }), "ab!");
    await user.click(screen.getByRole("button", { name: "Look up" }));

    expect(screen.getByRole("alert")).toHaveTextContent("Enter a valid player tag.");
    expect(onLookUp).not.toHaveBeenCalled();
  });
});

describe("Landing recents", () => {
  it("offers recent players and clears them", async () => {
    const user = userEvent.setup();
    saveRecentPlayer({ tag: "ABC123", name: "Ryan", window: 25 });
    render(<Landing onLookUp={vi.fn()} />);
    expect(screen.getByRole("link", { name: "Ryan #ABC123" })).toHaveAttribute(
      "href",
      "?tag=ABC123&window=25",
    );
    await user.click(screen.getByRole("button", { name: "Clear" }));
    expect(screen.queryByRole("navigation", { name: "Recent players" })).not.toBeInTheDocument();
    expect(readRecentPlayers()).toEqual([]);
  });
});

describe("TagForm", () => {
  it("starts from the current tag", () => {
    render(<TagForm initialTag="ABC123" onLookUp={vi.fn()} />);

    expect(screen.getByRole("textbox", { name: "Player tag" })).toHaveValue("ABC123");
  });
});
