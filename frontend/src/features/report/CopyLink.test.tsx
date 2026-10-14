import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { CopyLink } from "./CopyLink";

afterEach(() => {
  vi.unstubAllGlobals();
  Reflect.deleteProperty(document, "execCommand");
});

describe("CopyLink", () => {
  it.each(["unavailable", "rejected"])(
    "copies by selection when the Clipboard API is %s",
    async (failure) => {
      const writeText = vi.fn().mockRejectedValue(new Error("Clipboard denied"));
      vi.stubGlobal("navigator", {
        clipboard: failure === "unavailable" ? undefined : { writeText },
      });
      const copy = vi.fn(() => {
        const input = document.querySelector("textarea");
        expect(input).toHaveValue(window.location.href);
        expect(input?.selectionStart).toBe(0);
        expect(input?.selectionEnd).toBe(window.location.href.length);
        return true;
      });
      Object.defineProperty(document, "execCommand", { value: copy, configurable: true });
      render(<CopyLink />);
      const button = screen.getByRole("button", { name: "Copy link" });
      button.focus();
      fireEvent.click(button);

      expect(await screen.findByRole("button", { name: "Copied" })).toHaveFocus();
      expect(copy).toHaveBeenCalledWith("copy");
      expect(document.querySelector("textarea")).toBeNull();
    },
  );

  it("keeps the copy action available if selection copying fails", async () => {
    vi.stubGlobal("navigator", { clipboard: undefined });
    const copy = vi.fn(() => {
      throw new Error("Copy denied");
    });
    Object.defineProperty(document, "execCommand", { value: copy, configurable: true });
    render(<CopyLink />);
    fireEvent.click(screen.getByRole("button", { name: "Copy link" }));

    await waitFor(() => expect(copy).toHaveBeenCalledWith("copy"));
    expect(document.querySelector("textarea")).toBeNull();
    expect(screen.getByRole("button", { name: "Copy link" })).toBeEnabled();
  });
});
