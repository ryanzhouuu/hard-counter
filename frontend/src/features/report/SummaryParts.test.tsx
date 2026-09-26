import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ExpectationBar } from "./ExpectationBar";
import { WindowPicker } from "./WindowPicker";

describe("ExpectationBar", () => {
  it("labels a result above expectation in green", () => {
    render(<ExpectationBar actual={7} expected={5.4} total={10} />);

    expect(screen.getByText("Expected")).toBeInTheDocument();
    expect(screen.getByText("Above expected")).toHaveClass("gap-positive");
  });

  it("labels a result below expectation in red", () => {
    render(<ExpectationBar actual={2} expected={2.8} total={5} />);

    expect(screen.getByText("Won")).toBeInTheDocument();
    expect(screen.getByText("Below expected")).toHaveClass("gap-negative");
  });

  it("omits the gap label when the result matches expectation", () => {
    render(<ExpectationBar actual={3} expected={3} total={5} />);

    expect(screen.queryByText("Above expected")).not.toBeInTheDocument();
    expect(screen.queryByText("Below expected")).not.toBeInTheDocument();
  });
});

describe("WindowPicker", () => {
  it("marks the selected window and disables windows beyond the scored battles", async () => {
    const onSelect = vi.fn();
    render(<WindowPicker selected={10} available={12} onSelect={onSelect} />);

    expect(screen.getByRole("button", { name: "10 battles" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByRole("button", { name: "25 battles" })).toBeDisabled();
    await userEvent.click(screen.getByRole("button", { name: "5 battles" }));
    expect(onSelect).toHaveBeenCalledWith(5);
  });

  it("keeps the selected window enabled when it is larger than the scored battles", () => {
    render(<WindowPicker selected={25} available={12} onSelect={vi.fn()} />);

    expect(screen.getByRole("button", { name: "25 battles" })).toBeEnabled();
  });
});
