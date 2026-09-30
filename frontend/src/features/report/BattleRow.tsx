import { useId, useState } from "react";

import type { Battle, Outcome } from "../../api/playerAnalysis";
import { modeLabel, percent, relativeTime, skipLabel } from "../../lib/format";
import { type Upset, upsetOf } from "../../lib/upset";
import { DeckGrid, MiniDeck } from "./DeckGrid";
import "./battles.css";

const OUTCOMES: Record<Outcome, { letter: string; label: string }> = {
  win: { letter: "W", label: "Win" },
  loss: { letter: "L", label: "Loss" },
  draw: { letter: "D", label: "Draw" },
  unknown: { letter: "–", label: "Unknown result" },
};

function WinChance({ probability, upset }: { probability: number; upset: Upset | null }) {
  const value = percent(probability);
  return (
    <span className={upset ? `win-chance upset-${upset}` : "win-chance"}>
      <span className="win-chance-track" aria-hidden="true">
        <span style={{ width: value }} />
      </span>
      <span className="num">{value}</span>
      <span className="visually-hidden"> win chance{upset && `, upset ${upset}`}</span>
    </span>
  );
}

function detailMeta(battle: Battle, upset: Upset | null): string {
  const ago = relativeTime(battle.timestamp);
  const chance =
    battle.win_probability === null
      ? skipLabel(battle.skip_reason)
      : `your win chance ${percent(battle.win_probability)}`;
  const when = ago === "now" ? "just now" : ago && `${ago} ago`;
  return [modeLabel(battle.mode), when, chance, upset && `upset ${upset}`]
    .filter(Boolean)
    .join(" · ");
}

type BattleRowProps = { battle: Battle; repeatsDeck?: boolean };

function BattleRow({ battle, repeatsDeck = false }: BattleRowProps) {
  const [open, setOpen] = useState(false);
  const detailId = useId();
  const outcome = OUTCOMES[battle.outcome];
  const upset = upsetOf(battle);
  return (
    <li className="battle">
      <button
        type="button"
        className="battle-row"
        aria-expanded={open}
        aria-controls={detailId}
        onClick={() => setOpen((value) => !value)}
      >
        <span className={`outcome outcome-${battle.outcome}`}>
          <span aria-hidden="true">{outcome.letter}</span>
          <span className="visually-hidden">{outcome.label}</span>
        </span>
        <span className="battle-opponent">{battle.opponent_name}</span>
        <MiniDeck
          cards={battle.player_cards}
          className={repeatsDeck ? "mini-deck-player mini-deck-repeat" : "mini-deck-player"}
        />
        <span className="battle-vs" aria-hidden="true">
          vs
        </span>
        <MiniDeck cards={battle.opponent_cards} className="mini-deck-opponent" />
        {battle.win_probability === null ? (
          <span className="battle-skip">{skipLabel(battle.skip_reason)}</span>
        ) : (
          <WinChance probability={battle.win_probability} upset={upset} />
        )}
        <time className="battle-ago" dateTime={battle.timestamp ?? undefined}>
          {relativeTime(battle.timestamp)}
        </time>
        <span className="battle-chevron" aria-hidden="true" />
      </button>
      <div id={detailId} className="battle-detail" hidden={!open}>
        {open && (
          <>
            <div className="decks">
              <DeckGrid
                label="You"
                cards={battle.player_cards}
                tower={battle.player_tower}
                isPlayer
              />
              <DeckGrid
                label={battle.opponent_name}
                cards={battle.opponent_cards}
                tower={battle.opponent_tower}
              />
            </div>
            <p className="battle-meta">{detailMeta(battle, upset)}</p>
          </>
        )}
      </div>
    </li>
  );
}

export { BattleRow };
