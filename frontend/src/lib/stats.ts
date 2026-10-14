import type { Battle, Card } from "../api/playerAnalysis";

type ScoredBattle = Battle & { win_probability: number };
type FacedCard = { card: Card; faced: number; winChance: number };

function scoredWindow(battles: Battle[]): ScoredBattle[] {
  return battles.filter(
    (battle): battle is ScoredBattle => battle.in_window && battle.win_probability !== null,
  );
}

function winSpread(battles: Battle[]): number {
  const variance = scoredWindow(battles).reduce(
    (sum, { win_probability: p }) => sum + p * (1 - p),
    0,
  );
  return Math.sqrt(variance);
}

function facedMost(battles: Battle[], limit = 5): FacedCard[] {
  const tally = new Map<string, { card: Card; faced: number; chance: number }>();
  for (const battle of scoredWindow(battles)) {
    for (const card of battle.opponent_cards) {
      const entry = tally.get(card.name) ?? { card, faced: 0, chance: 0 };
      entry.faced += 1;
      entry.chance += battle.win_probability;
      tally.set(card.name, entry);
    }
  }
  return [...tally.values()]
    .map(({ card, faced, chance }) => ({ card, faced, winChance: chance / faced }))
    .sort((a, b) => b.faced - a.faced || a.card.name.localeCompare(b.card.name))
    .slice(0, limit);
}

export { facedMost, winSpread };
export type { FacedCard };
