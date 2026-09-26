import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { jsonResponse, makeAnalysis, makeSchedule } from "../test/fixtures";
import { App } from "./App";

const DISCLAIMER = "Unofficial fan project. Not affiliated with or endorsed by Supercell.";

beforeEach(() => {
  window.history.replaceState(null, "", "/");
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("App", () => {
  it("looks up a tag from the landing screen", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.resolve(jsonResponse(makeAnalysis()))),
    );
    const user = userEvent.setup();
    render(<App />);

    expect(screen.getByText(DISCLAIMER)).toBeInTheDocument();
    await user.type(screen.getByRole("textbox", { name: "Player tag" }), "#abc123");
    await user.click(screen.getByRole("button", { name: "Look up" }));

    expect(await screen.findByText("Won 7 of 10")).toBeInTheDocument();
    expect(window.location.search).toBe("?tag=ABC123&window=10");
  });

  it("shows the API message and recovers on retry", async () => {
    window.history.replaceState(null, "", "/?tag=ABC123");
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValueOnce(
          jsonResponse({ detail: "Clash Royale is unavailable. Try again shortly." }, 502),
        )
        .mockResolvedValueOnce(jsonResponse(makeAnalysis())),
    );
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByRole("alert")).toHaveTextContent("Clash Royale is unavailable.");
    await user.click(screen.getByRole("button", { name: "Retry" }));

    expect(await screen.findByText("Won 7 of 10")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("switches windows and expands a battle", async () => {
    window.history.replaceState(null, "", "/?tag=ABC123&window=10");
    const shortWindow = makeAnalysis({
      schedule: makeSchedule({
        requested_window: 5,
        actual_wins: 2,
        expected_wins: 2.8,
        performance_above_expectation: -0.8,
      }),
    });
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string) =>
        Promise.resolve(jsonResponse(url.endsWith("window=5") ? shortWindow : makeAnalysis())),
      ),
    );
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText("Won 7 of 10");

    await user.click(screen.getByRole("button", { name: "5 battles" }));
    expect(await screen.findByText("Won 2 of 5")).toBeInTheDocument();
    expect(screen.getByText("−0.8")).toHaveClass("tone-negative");

    await user.click(screen.getByRole("button", { name: /^Win/ }));
    expect(screen.getByRole("region", { name: "You deck" })).toBeVisible();
  });
});
