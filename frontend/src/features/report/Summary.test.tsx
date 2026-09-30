import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { makeAnalysis, makeBattle, makeSchedule } from "../../test/fixtures";
import { Summary } from "./Summary";

const insufficient = makeSchedule({
  status: "insufficient_data",
  eligible_count: 7,
  strength_of_schedule: null,
  expected_wins: null,
  actual_wins: null,
  performance_above_expectation: null,
});

function renderSummary(report = makeAnalysis()) {
  render(<Summary report={report} windowSize={10} onSelectWindow={() => undefined} />);
}

describe("Summary", () => {
  it("states the result in plain numbers", () => {
    renderSummary();

    expect(screen.getByRole("heading", { name: "Ryan" })).toBeInTheDocument();
    expect(screen.getByText("#ABC123 · 7,412 trophies")).toBeInTheDocument();
    expect(screen.getByText("Won 7 of 10")).toBeInTheDocument();
    expect(screen.getByText(/^Expected 5\.4/)).toHaveTextContent("Expected 5.4 · +1.6");
    expect(screen.getByText("+1.6")).toHaveClass("tone-positive");
    expect(screen.getByText("Avg win chance").parentElement).toHaveTextContent(
      "Avg win chance 54%",
    );
  });

  it("counts upsets inside the window", () => {
    renderSummary(
      makeAnalysis({
        battles: [
          makeBattle({ outcome: "win", win_probability: 0.3 }),
          makeBattle({ outcome: "win", win_probability: 0.35 }),
          makeBattle({ outcome: "loss", win_probability: 0.7 }),
          makeBattle({ outcome: "win", win_probability: 0.2, in_window: false }),
          makeBattle({ outcome: "win", win_probability: 0.62 }),
        ],
      }),
    );

    expect(screen.getByText("Upsets").parentElement).toHaveTextContent("Upsets 2 won · 1 lost");
  });

  it("explains disabled windows only when the report is available", () => {
    renderSummary(makeAnalysis({ schedule: makeSchedule({ eligible_count: 21 }) }));

    expect(screen.getByText("Only 21 scored battles")).toBeInTheDocument();
    expect(screen.getByRole("group", { name: "Battles counted" })).toHaveAccessibleDescription(
      "Only 21 scored battles",
    );
  });

  it("colors a result below expectation red", () => {
    renderSummary(
      makeAnalysis({
        schedule: makeSchedule({
          requested_window: 5,
          actual_wins: 2,
          expected_wins: 2.8,
          performance_above_expectation: -0.8,
        }),
      }),
    );

    expect(screen.getByText("Won 2 of 5")).toBeInTheDocument();
    expect(screen.getByText("−0.8")).toHaveClass("tone-negative");
  });

  it("omits missing trophies", () => {
    renderSummary(makeAnalysis({ player: { tag: "#ABC123", name: "Ryan", trophies: null } }));

    expect(screen.getByText("#ABC123")).toBeInTheDocument();
  });

  it("explains a window without enough scored battles", () => {
    renderSummary(makeAnalysis({ schedule: insufficient }));

    expect(screen.getAllByText("Only 7 scored battles")).toHaveLength(1);
    expect(screen.queryByText(/^Won/)).not.toBeInTheDocument();
  });

  it("explains an empty battle log", () => {
    renderSummary(makeAnalysis({ schedule: { ...insufficient, eligible_count: 0 }, battles: [] }));

    expect(screen.getByText("No recent battles.")).toBeInTheDocument();
  });
});
