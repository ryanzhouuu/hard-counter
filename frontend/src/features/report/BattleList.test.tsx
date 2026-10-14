import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { makeBattle } from "../../test/fixtures";
import { BattleList } from "./BattleList";

describe("BattleList", () => {
  it("lists window battles and reveals older and unscored ones on request", async () => {
    const user = userEvent.setup();
    render(
      <BattleList
        windowFilled
        battles={[
          makeBattle({ opponent_name: "Sora" }),
          makeBattle({ opponent_name: "Tariq", in_window: false }),
          makeBattle({ in_window: false, skip_reason: "draw", win_probability: null }),
        ]}
      />,
    );
    expect(screen.getAllByRole("listitem")).toHaveLength(1);

    await user.click(screen.getByRole("button", { name: "Show 1 older battle" }));
    expect(screen.getAllByRole("listitem")).toHaveLength(2);

    await user.click(screen.getByRole("button", { name: "Show 1 not scored" }));
    expect(screen.getAllByRole("listitem")).toHaveLength(3);

    await user.click(screen.getByRole("button", { name: "Hide 1 not scored" }));
    expect(screen.getAllByRole("listitem")).toHaveLength(2);
  });

  it("lists every scored battle when the window is not filled", () => {
    render(
      <BattleList
        windowFilled={false}
        battles={[makeBattle({ in_window: false }), makeBattle({ in_window: false })]}
      />,
    );

    expect(screen.getAllByRole("listitem")).toHaveLength(2);
    expect(screen.queryByRole("button", { name: /older/ })).not.toBeInTheDocument();
  });

  it("filters and sorts scored rows while leaving unscored ones alone", async () => {
    const user = userEvent.setup();
    render(
      <BattleList
        windowFilled={false}
        battles={[
          makeBattle({ opponent_name: "Ana", outcome: "win", win_probability: 0.7 }),
          makeBattle({ opponent_name: "Bo", outcome: "loss", win_probability: 0.4 }),
          makeBattle({ opponent_name: "Cy", outcome: "win", win_probability: 0.2 }),
          makeBattle({ opponent_name: "Dee", skip_reason: "draw", win_probability: null }),
        ]}
      />,
    );
    await user.click(screen.getByRole("button", { name: "Wins" }));
    expect(screen.getByRole("button", { name: "Wins" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.queryByText("Bo")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Win chance" }));
    const opponents = () =>
      screen
        .getAllByRole("listitem")
        .map((row) => row.querySelector(".battle-opponent")?.textContent);
    expect(opponents()).toEqual(["Cy", "Ana"]);
    await user.click(screen.getByRole("button", { name: "Upsets" }));
    expect(opponents()).toEqual(["Cy"]);
    await user.click(screen.getByRole("button", { name: "Show 1 not scored" }));
    expect(screen.getByText("Dee")).toBeInTheDocument();
  });

  it("says when no battles match a filter", async () => {
    const user = userEvent.setup();
    render(<BattleList windowFilled={false} battles={[makeBattle({ outcome: "win" })]} />);
    await user.click(screen.getByRole("button", { name: "Losses" }));
    expect(screen.getByText("No battles match.")).toBeInTheDocument();
  });

  it("counts only matching older battles and hides the toggle when none match", async () => {
    const user = userEvent.setup();
    render(
      <BattleList
        windowFilled
        battles={[
          makeBattle(),
          makeBattle({ in_window: false, outcome: "win", opponent_name: "Ana" }),
          makeBattle({
            in_window: false,
            outcome: "loss",
            win_probability: 0.4,
            opponent_name: "Bo",
          }),
        ]}
      />,
    );
    expect(screen.getByRole("button", { name: "Show 2 older battles" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Wins" }));
    await user.click(screen.getByRole("button", { name: "Show 1 older battle" }));
    expect(screen.getByText("Ana")).toBeInTheDocument();
    expect(screen.queryByText("Bo")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Hide 1 older battle" }));
    await user.click(screen.getByRole("button", { name: "Upsets" }));
    expect(screen.queryByRole("button", { name: /older battle/ })).not.toBeInTheDocument();
  });
});
