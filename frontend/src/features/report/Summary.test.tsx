import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { makeAnalysis, makeSchedule } from "../../test/fixtures";
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

    expect(screen.getByText("Only 7 scored battles")).toBeInTheDocument();
    expect(screen.queryByText(/^Won/)).not.toBeInTheDocument();
  });

  it("explains an empty battle log", () => {
    renderSummary(makeAnalysis({ schedule: { ...insufficient, eligible_count: 0 }, battles: [] }));

    expect(screen.getByText("No recent battles.")).toBeInTheDocument();
  });
});
