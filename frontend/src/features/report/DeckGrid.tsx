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

type DeckGridProps = {
  label: string;
  cards: Card[];
  tower?: Tower | null;
  isPlayer?: boolean;
  mirrored?: boolean;
};

function TowerTile({ tower }: { tower: Tower }) {
  return (
    <figure className="tower-tile">
      <CardImage card={tower} decorative />
      <figcaption>
        {tower.name}
        {tower.level !== null && <span className="num"> · Lv {tower.level}</span>}
      </figcaption>
    </figure>
  );
}

function DeckGrid({
  label,
  cards,
  tower = null,
  isPlayer = false,
  mirrored = false,
}: DeckGridProps) {
  return (
    <section className={mirrored ? "deck deck-mirrored" : "deck"} aria-label={`${label} deck`}>
      <h3 className={isPlayer ? "deck-label deck-label-player" : "deck-label"}>{label}</h3>
      <div className="deck-side">
        {tower && <TowerTile tower={tower} />}
        <div className="deck-grid">
          {cards.map((card, index) => (
            <CardImage key={`${card.name}-${index}`} card={card} />
          ))}
        </div>
      </div>
    </section>
  );
}

export { DeckGrid, MiniDeck };
