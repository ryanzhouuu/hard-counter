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

  it("marks an upset win in the row and the details", async () => {
    const user = userEvent.setup();
    render(
      <ul>
        <BattleRow battle={makeBattle({ win_probability: 0.3, timestamp: null })} />
      </ul>,
    );
    const row = screen.getByRole("button", { name: /win chance, upset win/ });
    expect(row.querySelector(".win-chance")).toHaveClass("upset-win");

    await user.click(row);

    expect(screen.getByText("Ranked · your win chance 30% · upset win")).toBeInTheDocument();
  });

  it("shows tower troops in the details", async () => {
    const user = userEvent.setup();
    const tower = { name: "Cannoneer", icon_url: null, identity: "cannoneer:tower", level: 11 };
    render(
      <ul>
        <BattleRow battle={makeBattle({ player_tower: tower, opponent_tower: null })} />
      </ul>,
    );

    await user.click(screen.getByRole("button", { name: /^Win/ }));

    const deck = screen.getByRole("region", { name: "You deck" });
    expect(deck.querySelector(".deck-tower")).toHaveTextContent("Cannoneer");
    expect(
      screen.getByRole("region", { name: "Rival deck" }).querySelector(".deck-tower"),
    ).toBeNull();
  });

  it("fades a deck repeated from the previous row", () => {
    const { container } = render(
      <ul>
        <BattleRow battle={makeBattle()} repeatsDeck />
      </ul>,
    );

    expect(container.querySelector(".mini-deck-player")).toHaveClass("mini-deck-repeat");
    expect(container.querySelector(".mini-deck-opponent")).not.toHaveClass("mini-deck-repeat");
  });
});
