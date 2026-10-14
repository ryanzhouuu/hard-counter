import type { Battle } from "../api/playerAnalysis";
import { upsetOf } from "./upset";

const BATTLE_FILTERS = ["all", "wins", "losses", "upsets"] as const;
const BATTLE_SORTS = ["recent", "chance"] as const;
type BattleFilter = (typeof BATTLE_FILTERS)[number];
type BattleSort = (typeof BATTLE_SORTS)[number];

const FILTER_LABELS: Record<BattleFilter, string> = {
  all: "All",
  wins: "Wins",
  losses: "Losses",
  upsets: "Upsets",
};
const SORT_LABELS: Record<BattleSort, string> = { recent: "Recent", chance: "Win chance" };

function matches(battle: Battle, filter: BattleFilter): boolean {
  if (filter === "wins") return battle.outcome === "win";
  if (filter === "losses") return battle.outcome === "loss";
  if (filter === "upsets") return upsetOf(battle) !== null;
  return true;
}

function arrangeBattles<T extends { battle: Battle }>(
  items: T[],
  filter: BattleFilter,
  sort: BattleSort,
): T[] {
  const kept = items.filter(({ battle }) => matches(battle, filter));
  if (sort === "recent") return kept;
  return kept.sort((a, b) => (a.battle.win_probability ?? 1) - (b.battle.win_probability ?? 1));
}

export { BATTLE_FILTERS, BATTLE_SORTS, FILTER_LABELS, SORT_LABELS, arrangeBattles };
export type { BattleFilter, BattleSort };
