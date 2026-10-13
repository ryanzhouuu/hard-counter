import { useId, useState } from "react";

import type { CardChoice, MatchupDeck, TowerChoice } from "../../api/deckMatchup";
import "./deck-builder.css";

type DeckBuilderProps = {
  label: string;
  cards: CardChoice[];
  towers: TowerChoice[];
  deck: MatchupDeck;
  onChange: (deck: MatchupDeck) => void;
};

const FORMS: Record<string, string> = {
  base: "Base",
  evolution: "Evolution",
  hero: "Hero",
  champion: "Champion",
};

function DeckBuilder({ label, cards, towers, deck, onChange }: DeckBuilderProps) {
  const id = useId();
  const [search, setSearch] = useState("");
  const selectedRoots = new Set(deck.cards.map((identity) => identity.split(":")[0]));
  const matches = cards.filter((card) =>
    `${card.name} ${FORMS[card.form]}`.toLowerCase().includes(search.trim().toLowerCase()),
  );

  return (
    <fieldset className="deck-builder">
      <legend>
        {label} <span className="deck-count num">{deck.cards.length} / 8</span>
      </legend>
      <ol className="deck-slots" aria-label={`${label} selected cards`}>
        {Array.from({ length: 8 }, (_, index) => {
          const identity = deck.cards[index];
          const card = cards.find((choice) => choice.identity === identity);
          return (
            <li key={index}>
              {card ? (
                <button
                  type="button"
                  aria-label={`Remove ${card.name} from ${label}`}
                  onClick={() =>
                    onChange({ ...deck, cards: deck.cards.filter((value) => value !== identity) })
                  }
                >
                  <strong>{card.name}</strong>
                  <small>
                    {FORMS[card.form]} <span aria-hidden="true">×</span>
                  </small>
                </button>
              ) : (
                <span className="empty-slot" aria-label={`Empty slot ${index + 1}`}>
                  {String(index + 1).padStart(2, "0")}
                </span>
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
        {matches.map((card) => (
          <button
            key={card.identity}
            type="button"
            aria-label={`Add ${card.name} to ${label}`}
            disabled={deck.cards.length === 8 || selectedRoots.has(card.identity.split(":")[0])}
            onClick={() => {
              onChange({ ...deck, cards: [...deck.cards, card.identity] });
              setSearch("");
            }}
          >
            <span>{card.name}</span>
            <small>{FORMS[card.form]}</small>
          </button>
        ))}
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
