import type { Card, Tower } from "../../api/playerAnalysis";
import { CardImage } from "./CardImage";

function MiniDeck({ cards, className }: { cards: Card[]; className?: string }) {
  return (
    <span className={className ? `mini-deck ${className}` : "mini-deck"} aria-hidden="true">
      {cards.map((card, index) => (
        <CardImage key={`${card.name}-${index}`} card={card} decorative />
      ))}
    </span>
  );
}

type DeckGridProps = { label: string; cards: Card[]; tower?: Tower | null; isPlayer?: boolean };

function DeckGrid({ label, cards, tower = null, isPlayer = false }: DeckGridProps) {
  return (
    <section className="deck" aria-label={`${label} deck`}>
      <h3 className={isPlayer ? "deck-label deck-label-player" : "deck-label"}>{label}</h3>
      <div className="deck-grid">
        {cards.map((card, index) => (
          <CardImage key={`${card.name}-${index}`} card={card} />
        ))}
      </div>
      {tower && (
        <p className="deck-tower">
          <span className="deck-tower-icon">
            <CardImage card={tower} decorative />
          </span>
          {tower.name}
        </p>
      )}
    </section>
  );
}

export { DeckGrid, MiniDeck };
