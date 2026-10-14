import { useId, useState } from "react";

import type { CardChoice, MatchupDeck, TowerChoice } from "../../api/deckMatchup";
import { SLOT_LABELS, averageElixir, blockReason, slotLayout } from "../../lib/deckRules";
import { CardImage } from "../report/CardImage";
import "./deck-builder.css";

type DeckBuilderProps = {
  label: string;
  cards: CardChoice[];
  towers: TowerChoice[];
  deck: MatchupDeck;
  onChange: (deck: MatchupDeck) => void;
  onUseLatest?: () => void;
  note?: string | null;
};

const FORMS: Record<string, string> = {
  base: "Base",
  evolution: "Evolution",
  hero: "Hero",
  champion: "Champion",
};

function DeckBuilder({
  label,
  cards,
  towers,
  deck,
  onChange,
  onUseLatest,
  note,
}: DeckBuilderProps) {
  const id = useId();
  const [search, setSearch] = useState("");
  const byIdentity = new Map(cards.map((card) => [card.identity, card]));
  const chosen = deck.cards.flatMap((identity) => byIdentity.get(identity) ?? []);
  const elixir = averageElixir(chosen);
  const matches = cards.filter((card) =>
    `${card.name} ${FORMS[card.form]}`.toLowerCase().includes(search.trim().toLowerCase()),
  );

  return (
    <fieldset className="deck-builder">
      <legend>
        {label}{" "}
        <span className="deck-count num">
          {deck.cards.length} / 8{elixir !== null && ` · ${elixir.toFixed(1)} elixir`}
        </span>
      </legend>
      <div className="deck-actions">
        {onUseLatest && (
          <button type="button" onClick={onUseLatest}>
            Use my latest
          </button>
        )}
        <button
          type="button"
          aria-label={`Clear ${label}`}
          onClick={() => onChange({ cards: [], tower: null })}
        >
          Clear
        </button>
      </div>
      {note && (
        <p className="deck-note" role="status">
          {note}
        </p>
      )}
      <ol className="deck-slots" aria-label={`${label} selected cards`}>
        {slotLayout(deck.cards).map((identity, index) => {
          const card = identity === null ? undefined : byIdentity.get(identity);
          const slot = SLOT_LABELS[index];
          return (
            <li key={index}>
              {slot && <span className="slot-label">{slot}</span>}
              {card ? (
                <button
                  type="button"
                  aria-label={`Remove ${card.name} from ${label}`}
                  title={card.name}
                  onClick={() =>
                    onChange({
                      ...deck,
                      cards: deck.cards.filter((value) => value !== card.identity),
                    })
                  }
                >
                  <CardImage card={card} decorative />
                </button>
              ) : (
                <span className="empty-slot" aria-label={`Empty slot ${index + 1}`} />
              )}
            </li>
          );
        })}
      </ol>
      <label htmlFor={`${id}-search`}>Search cards</label>
      <input
        id={`${id}-search`}
        type="search"
        aria-label={`${label} search cards`}
        placeholder={deck.cards.length === 8 ? "Deck complete" : "Card name or form…"}
        value={search}
        disabled={deck.cards.length === 8}
        onChange={(event) => setSearch(event.target.value)}
      />
      <div
        className="card-choices"
        role="group"
        aria-label={`${label} card choices`}
        hidden={deck.cards.length === 8}
      >
        {matches.map((card) => {
          const reason = blockReason(deck.cards, card.identity);
          return (
            <button
              key={card.identity}
              type="button"
              aria-label={`Add ${card.name} to ${label}`}
              aria-disabled={reason !== null || undefined}
              title={reason ?? undefined}
              onClick={() => {
                if (reason !== null) return;
                onChange({ ...deck, cards: [...deck.cards, card.identity] });
                setSearch("");
              }}
            >
              <CardImage card={card} decorative />
              <span>{card.name}</span>
              {card.form !== "base" && <small>{FORMS[card.form]}</small>}
            </button>
          );
        })}
        {matches.length === 0 && <p role="status">No matching cards.</p>}
      </div>
      {towers.length > 0 && (
        <>
          <label htmlFor={`${id}-tower`}>Tower troop</label>
          <select
            id={`${id}-tower`}
            aria-label={`${label} tower troop`}
            value={deck.tower ?? ""}
            onChange={(event) => onChange({ ...deck, tower: event.target.value || null })}
          >
            <option value="">Select a tower troop</option>
            {towers.map((tower) => (
              <option key={tower.identity} value={tower.identity}>
                {tower.name}
              </option>
            ))}
          </select>
        </>
      )}
    </fieldset>
  );
}

export { DeckBuilder };
