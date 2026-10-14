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
    expect(screen.getByText("Ranked", { selector: ".battle-meta" })).toBeInTheDocument();
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

    expect(screen.getByText("Ranked · upset win")).toBeInTheDocument();
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
    expect(deck.querySelector(".tower-tile")).toHaveTextContent("Cannoneer · Lv 11");
    expect(
      screen.getByRole("region", { name: "Rival deck" }).querySelector(".tower-tile"),
    ).toBeNull();
  });

  it("shows the full battle time on hover", () => {
    const { container } = render(
      <ul>
        <BattleRow battle={makeBattle()} />
      </ul>,
    );
    expect(container.querySelector("time")).toHaveAttribute("title", expect.stringMatching(/2026/));
  });

  it("grows the row into the detail with a full chance bar", async () => {
    const user = userEvent.setup();
    render(
      <ul>
        <BattleRow battle={makeBattle()} />
      </ul>,
    );
    await user.click(screen.getByRole("button", { name: /^Win/ }));
    expect(screen.getByRole("listitem")).toHaveClass("battle-open");
    const meter = screen.getByRole("meter", { name: "Your win chance" });
    expect(meter).toHaveAttribute("aria-valuenow", "62");
    expect(screen.getByText("You 62%")).toBeInTheDocument();
    expect(screen.getByText("38% Rival")).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Rival deck" })).toHaveClass("deck-mirrored");
  });

  it("shows the skip reason instead of a bar for unscored battles", async () => {
    const user = userEvent.setup();
    render(
      <ul>
        <BattleRow
          battle={makeBattle({ outcome: "draw", skip_reason: "draw", win_probability: null })}
        />
      </ul>,
    );
    await user.click(screen.getByRole("button", { name: /^Draw/ }));
    expect(screen.queryByRole("meter")).not.toBeInTheDocument();
    expect(screen.getByText("Draw", { selector: ".battle-skip-note" })).toBeInTheDocument();
  });
});
