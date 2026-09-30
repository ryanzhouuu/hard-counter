import { useState } from "react";

import type { Battle, Card } from "../../api/playerAnalysis";
import { BattleRow } from "./BattleRow";
import "./battle-list.css";

type KeyedBattle = { key: string; battle: Battle };
type BattleListProps = { battles: Battle[]; windowFilled: boolean };

function deckKey(cards: Card[]): string {
  return cards
    .map((card) => card.name)
    .sort()
    .join("|");
}

function Rows({ items }: { items: KeyedBattle[] }) {
  return (
    <ul className="battle-rows">
      {items.map(({ key, battle }, index) => (
        <BattleRow
          key={key}
          battle={battle}
          repeatsDeck={
            index > 0 &&
            deckKey(items[index - 1].battle.player_cards) === deckKey(battle.player_cards)
          }
        />
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
  if (battles.length === 0) return null;

  const keyed = battles.map((battle, index) => ({
    key: `${battle.timestamp ?? "unknown"}-${index}`,
    battle,
  }));
  const scored = keyed.filter(({ battle }) => battle.skip_reason === null);
  const skipped = keyed.filter(({ battle }) => battle.skip_reason !== null);
  const primary = windowFilled ? scored.filter(({ battle }) => battle.in_window) : scored;
  const older = windowFilled ? scored.filter(({ battle }) => !battle.in_window) : [];

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
              {showOlder ? "Hide" : "Show"} {plural(older.length, "older battle")}
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
