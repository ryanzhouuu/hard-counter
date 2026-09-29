import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { Landing } from "./Landing";
import { TagForm } from "./TagForm";

describe("Landing", () => {
  it("submits a normalized tag", async () => {
    const user = userEvent.setup();
    const onLookUp = vi.fn();
    render(<Landing onLookUp={onLookUp} />);

    expect(screen.getByRole("heading", { name: "Hard Counter" })).toBeInTheDocument();
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

describe("TagForm", () => {
  it("starts from the current tag", () => {
    render(<TagForm initialTag="ABC123" onLookUp={vi.fn()} />);

    expect(screen.getByRole("textbox", { name: "Player tag" })).toHaveValue("ABC123");
  });
});
