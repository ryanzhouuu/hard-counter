import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { App } from "./App";

describe("App", () => {
  it("renders the first-run analysis overview", () => {
    render(<App />);

    expect(screen.getByRole("heading", { name: /read the matchup/i })).toBeInTheDocument();
    expect(screen.getByText("Matchup probability")).toBeInTheDocument();
    expect(screen.getByText("Your analysis desk is ready for data.")).toBeInTheDocument();
    expect(screen.getByText("Local workspace")).toBeInTheDocument();
  });

  it("switches to a scaffolded workspace and returns to overview", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: "Matchups" }));
    expect(screen.getByRole("heading", { name: "Matchups" })).toBeInTheDocument();
    expect(
      screen.getByText(
        "The analysis surface is scaffolded and waiting for its first data pipeline.",
      ),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /return to overview/i }));
    expect(screen.getByRole("heading", { name: /read the matchup/i })).toBeInTheDocument();
  });
});
