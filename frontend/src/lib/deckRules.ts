import type { CardChoice } from "../api/deckMatchup";

type Special = "evolution" | "hero" | null;

const SLOT_LABELS = ["EVO", "HERO", "WILD", null, null, null, null, null] as const;
const PLAIN_FIRST = [3, 4, 5, 6, 7, 0, 1, 2];

function rootOf(identity: string): string {
  return identity.slice(0, identity.lastIndexOf(":"));
}

function specialOf(identity: string): Special {
  const form = identity.slice(identity.lastIndexOf(":") + 1);
  if (form === "evolution") return "evolution";
  return form === "hero" || form === "champion" ? "hero" : null;
}

function slotLayout(cards: string[]): (string | null)[] {
  const slots: (string | null)[] = Array<string | null>(8).fill(null);
  const evolutions = cards.filter((card) => specialOf(card) === "evolution");
  const heroes = cards.filter((card) => specialOf(card) === "hero");
  slots[0] = evolutions[0] ?? null;
  slots[1] = heroes[0] ?? null;
  slots[2] = evolutions[1] ?? heroes[1] ?? null;
  const open = PLAIN_FIRST.filter((index) => slots[index] === null);
  cards
    .filter((card) => specialOf(card) === null)
    .forEach((card, index) => {
      slots[open[index]] = card;
    });
  return slots;
}

function blockReason(cards: string[], identity: string): string | null {
  if (cards.some((card) => rootOf(card) === rootOf(identity))) return "Already in deck";
  if (cards.length >= 8) return "Deck complete";
  const special = specialOf(identity);
  if (special === null) return null;
  const evolutions = cards.filter((card) => specialOf(card) === "evolution").length;
  const heroes = cards.filter((card) => specialOf(card) === "hero").length;
  if (special === "evolution" && evolutions >= 2) return "Evolution slots full";
  if (special === "hero" && heroes >= 2) return "Hero slots full";
  if (evolutions + heroes >= 3) return "No special slots left";
  return null;
}

function averageElixir(cards: CardChoice[]): number | null {
  const costs = cards.flatMap((card) => (card.elixir === null ? [] : [card.elixir]));
  return costs.length === 0 ? null : costs.reduce((sum, cost) => sum + cost, 0) / costs.length;
}

export { SLOT_LABELS, averageElixir, blockReason, rootOf, slotLayout };
