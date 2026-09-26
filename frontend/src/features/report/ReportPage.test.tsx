import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ComponentProps } from "react";
import { describe, expect, it, vi } from "vitest";

import { AnalysisError } from "../../api/playerAnalysis";
import { makeAnalysis } from "../../test/fixtures";
import { ReportPage } from "./ReportPage";

function renderPage(props: Partial<ComponentProps<typeof ReportPage>> = {}) {
  render(
    <ReportPage
      tag="ABC123"
      windowSize={10}
      report={null}
      loading={false}
      error={null}
      onLookUp={vi.fn()}
      onSelectWindow={vi.fn()}
      onRetry={vi.fn()}
      {...props}
    />,
  );
}

describe("ReportPage", () => {
  it("shows a skeleton while the first report loads", () => {
    renderPage({ loading: true });

    expect(screen.getByLabelText("Loading report")).toHaveAttribute("aria-busy", "true");
    expect(screen.getByRole("textbox", { name: "Player tag" })).toHaveValue("ABC123");
  });

  it("keeps the report visible while a new window loads", () => {
    renderPage({ report: makeAnalysis(), loading: true });

    expect(screen.getByText("Won 7 of 10")).toBeInTheDocument();
    expect(screen.getByRole("main")).toHaveAttribute("aria-busy", "true");
  });

  it("offers a retry for temporary failures", async () => {
    const onRetry = vi.fn();
    renderPage({
      error: new AnalysisError("Clash Royale is unavailable. Try again shortly.", 502),
      onRetry,
    });

    expect(screen.getByRole("alert")).toHaveTextContent("Clash Royale is unavailable.");
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(onRetry).toHaveBeenCalled();
  });

  it("does not offer a retry for permanent failures", () => {
    renderPage({ error: new AnalysisError("Player tag not found.", 404) });

    expect(screen.getByRole("alert")).toHaveTextContent("Player tag not found.");
    expect(screen.queryByRole("button", { name: "Retry" })).not.toBeInTheDocument();
  });
});
