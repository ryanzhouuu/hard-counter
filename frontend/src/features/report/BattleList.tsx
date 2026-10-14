import { useState } from "react";

import type { Battle } from "../../api/playerAnalysis";
import {
  BATTLE_FILTERS,
  BATTLE_SORTS,
  FILTER_LABELS,
  SORT_LABELS,
  arrangeBattles,
  type BattleFilter,
  type BattleSort,
} from "../../lib/battleView";
import { BattleRow } from "./BattleRow";
import "./battle-list.css";

type KeyedBattle = { key: string; battle: Battle };
type BattleListProps = { battles: Battle[]; windowFilled: boolean };

function Rows({ items }: { items: KeyedBattle[] }) {
  return (
    <ul className="battle-rows">
      {items.map(({ key, battle }) => (
        <BattleRow key={key} battle={battle} />
      ))}
    </ul>
  );
}

function plural(count: number, noun: string): string {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

function BattleList({ battles, windowFilled }: BattleListProps) {
  const [showOlder, setShowOlder] = useState(false);
  const [showSkipped, setShowSkipped] = useState(false);
  const [filter, setFilter] = useState<BattleFilter>("all");
  const [sort, setSort] = useState<BattleSort>("recent");
  if (battles.length === 0) return null;

  const keyed = battles.map((battle, index) => ({
    key: `${battle.timestamp ?? "unknown"}-${index}`,
    battle,
  }));
  const scored = keyed.filter(({ battle }) => battle.skip_reason === null);
  const skipped = keyed.filter(({ battle }) => battle.skip_reason !== null);
  const olderAll = windowFilled ? scored.filter(({ battle }) => !battle.in_window) : [];
  const primary = arrangeBattles(
    windowFilled ? scored.filter(({ battle }) => battle.in_window) : scored,
    filter,
    sort,
  );
  const older = arrangeBattles(olderAll, filter, sort);

  return (
    <section className="battle-list" aria-label="Recent battles">
      <div className="battle-tools">
        <div role="group" aria-label="Filter battles">
          {BATTLE_FILTERS.map((value) => (
            <button
              key={value}
              type="button"
              aria-pressed={filter === value}
              onClick={() => setFilter(value)}
            >
              {FILTER_LABELS[value]}
            </button>
          ))}
        </div>
        <div role="group" aria-label="Sort battles">
          {BATTLE_SORTS.map((value) => (
            <button
              key={value}
              type="button"
              aria-pressed={sort === value}
              onClick={() => setSort(value)}
            >
              {SORT_LABELS[value]}
            </button>
          ))}
        </div>
      </div>
      {primary.length === 0 && (!showOlder || older.length === 0) && (
        <p className="battle-empty">No battles match.</p>
      )}
      <Rows items={primary} />
      {showOlder && <Rows items={older} />}
      {showSkipped && <Rows items={skipped} />}
      {(olderAll.length > 0 || skipped.length > 0) && (
        <div className="battle-more">
          {olderAll.length > 0 && (
            <button
              type="button"
              aria-expanded={showOlder}
              onClick={() => setShowOlder((value) => !value)}
            >
              {showOlder ? "Hide" : "Show"} {plural(olderAll.length, "older battle")}
            </button>
          )}
          {skipped.length > 0 && (
            <button
              type="button"
              aria-expanded={showSkipped}
              onClick={() => setShowSkipped((value) => !value)}
            >
              {showSkipped ? "Hide" : "Show"} {skipped.length} not scored
            </button>
          )}
        </div>
      )}
    </section>
  );
}

export { BattleList };
