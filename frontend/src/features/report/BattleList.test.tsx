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

    await user.click(screen.getByRole("button", { name: "1 not scored" }));
    expect(screen.getAllByRole("listitem")).toHaveLength(3);
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
});
