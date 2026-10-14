import type { MatchupDeck, MatchupOptions } from "../../api/deckMatchup";
import type { PlayerAnalysis } from "../../api/playerAnalysis";

type LatestDeck = { deck: MatchupDeck } | { message: string };

function latestRankedDeck(report: PlayerAnalysis, options: MatchupOptions): LatestDeck {
  const battle = report.battles.find((entry) => entry.mode.startsWith("Ranked1v1"));
  if (!battle) return { message: "No recent ranked battle." };
  const known = new Set(options.cards.map((card) => card.identity));
  const cards = battle.player_cards.flatMap((card) =>
    card.identity !== null && known.has(card.identity) ? [card.identity] : [],
  );
  if (cards.length !== 8) {
    return { message: "Your latest ranked deck has cards this model doesn't support." };
  }
  if (options.model.input_scope === "deck_only") return { deck: { cards, tower: null } };
  const tower = battle.player_tower?.identity ?? null;
  if (tower === null || !options.towers.some((choice) => choice.identity === tower)) {
    return { message: "Your latest ranked tower troop isn't supported." };
  }
  return { deck: { cards, tower } };
}

export { latestRankedDeck };
