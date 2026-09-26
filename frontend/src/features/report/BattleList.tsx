import { useState } from "react";

import type { Battle } from "../../api/playerAnalysis";
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

function BattleList({ battles, windowFilled }: BattleListProps) {
  const [showOlder, setShowOlder] = useState(false);
  const [showSkipped, setShowSkipped] = useState(false);
  if (battles.length === 0) return null;

  const keyed = battles.map((battle, index) => ({
    key: `${battle.timestamp ?? "unknown"}-${index}`,
    battle,
  }));
  const scored = keyed.filter(({ battle }) => battle.skip_reason === null);
  const skipped = keyed.filter(({ battle }) => battle.skip_reason !== null);
  const primary = windowFilled ? scored.filter(({ battle }) => battle.in_window) : scored;
  const older = windowFilled ? scored.filter(({ battle }) => !battle.in_window) : [];
  const olderLabel = `${older.length} older ${older.length === 1 ? "battle" : "battles"}`;

  return (
    <section className="battle-list" aria-label="Recent battles">
      <Rows items={primary} />
      {showOlder && <Rows items={older} />}
      {showSkipped && <Rows items={skipped} />}
      {(older.length > 0 || skipped.length > 0) && (
        <div className="battle-more">
          {older.length > 0 && (
            <button
              type="button"
              aria-expanded={showOlder}
              onClick={() => setShowOlder((value) => !value)}
            >
              {showOlder ? "Hide" : "Show"} {olderLabel}
            </button>
          )}
          {skipped.length > 0 && (
            <button
              type="button"
              aria-expanded={showSkipped}
              onClick={() => setShowSkipped((value) => !value)}
            >
              {skipped.length} not scored
            </button>
          )}
        </div>
      )}
    </section>
  );
}

export { BattleList };
