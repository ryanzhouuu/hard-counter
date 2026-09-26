import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { makeBattle } from "../../test/fixtures";
import { BattleRow } from "./BattleRow";

describe("BattleRow", () => {
  it("expands to show both decks", async () => {
    const user = userEvent.setup();
    render(
      <ul>
        <BattleRow battle={makeBattle({ opponent_name: "Sora", timestamp: null })} />
      </ul>,
    );
    const row = screen.getByRole("button", { name: /^Win/ });
    expect(row).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("region", { name: "You deck" })).not.toBeInTheDocument();

    await user.click(row);

    expect(row).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByRole("region", { name: "You deck" })).toBeVisible();
    expect(screen.getByRole("region", { name: "Sora deck" })).toBeVisible();
    expect(screen.getByText("Ranked · your win chance 62%")).toBeInTheDocument();
  });

  it("shows why an unscored battle has no win chance", () => {
    render(
      <ul>
        <BattleRow
          battle={makeBattle({ outcome: "draw", skip_reason: "draw", win_probability: null })}
        />
      </ul>,
    );

    expect(screen.getByText("Draw", { selector: ".battle-skip" })).toBeInTheDocument();
    expect(screen.queryByText(/%/)).not.toBeInTheDocument();
  });
});
