import type { Card } from "../../api/playerAnalysis";
import { CardImage } from "./CardImage";

function MiniDeck({ cards }: { cards: Card[] }) {
  return (
    <span className="mini-deck" aria-hidden="true">
      {cards.map((card, index) => (
        <CardImage key={`${card.name}-${index}`} card={card} decorative />
      ))}
    </span>
  );
}

type DeckGridProps = { label: string; cards: Card[]; isPlayer?: boolean };

function DeckGrid({ label, cards, isPlayer = false }: DeckGridProps) {
  return (
    <section className="deck" aria-label={`${label} deck`}>
      <h3 className={isPlayer ? "deck-label deck-label-player" : "deck-label"}>{label}</h3>
      <div className="deck-grid">
        {cards.map((card, index) => (
          <CardImage key={`${card.name}-${index}`} card={card} />
        ))}
      </div>
    </section>
  );
}

export { DeckGrid, MiniDeck };
