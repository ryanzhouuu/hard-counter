import type { Battle } from "../api/playerAnalysis";

const UPSET_WIN_MAX = 0.4;
const UPSET_LOSS_MIN = 0.6;
const UPSET_DEFINITION = "Wins below 40% or losses above 60% win chance";

type Upset = "win" | "loss";

function upsetOf({ outcome, win_probability }: Battle): Upset | null {
  if (win_probability === null) return null;
  if (outcome === "win" && win_probability < UPSET_WIN_MAX) return "win";
  if (outcome === "loss" && win_probability > UPSET_LOSS_MIN) return "loss";
  return null;
}

export { UPSET_DEFINITION, upsetOf };
export type { Upset };
