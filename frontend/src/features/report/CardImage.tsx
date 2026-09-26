import { useState } from "react";

import type { Card } from "../../api/playerAnalysis";

type CardImageProps = { card: Card; decorative?: boolean };

function CardImage({ card, decorative = false }: CardImageProps) {
  const [failed, setFailed] = useState(false);
  if (card.icon_url === null || failed) {
    return (
      <span className="card-tile" title={card.name} aria-hidden={decorative || undefined}>
        {card.name}
      </span>
    );
  }
  return (
    <img
      className="card-image"
      src={card.icon_url}
      alt={decorative ? "" : card.name}
      loading="lazy"
      onError={() => setFailed(true)}
    />
  );
}

export { CardImage };
