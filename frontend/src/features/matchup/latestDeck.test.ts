import { describe, expect, it } from "vitest";

import { cards, makeAnalysis, makeBattle } from "../../test/fixtures";
import { matchupOptions } from "../../test/matchupFixtures";
import { latestRankedDeck } from "./latestDeck";

const names = ["Knight", "Archers", "Goblins", "Giant", "Minions", "Balloon", "Witch", "Bomber"];
const identities = names.map((name) => `${name.toLowerCase()}:base`);
const tower = { name: "Cannoneer", icon_url: null, identity: "cannoneer:tower", level: 11 };

describe("latestRankedDeck", () => {
  it("takes the newest ranked deck and tower", () => {
    const report = makeAnalysis({
      battles: [
        makeBattle({ mode: "Ladder", player_cards: cards(["Golem"]) }),
        makeBattle({ player_cards: cards(names), player_tower: tower }),
      ],
    });
    expect(latestRankedDeck(report, matchupOptions)).toEqual({
      deck: { cards: identities, tower: "cannoneer:tower" },
    });
  });

  it("sends no tower to a cards-only model", () => {
    const report = makeAnalysis({
      battles: [makeBattle({ player_cards: cards(names), player_tower: tower })],
    });
    const cardsOnly = {
      ...matchupOptions,
      model: { ...matchupOptions.model, input_scope: "deck_only" as const },
      towers: [],
    };
    expect(latestRankedDeck(report, cardsOnly)).toEqual({
      deck: { cards: identities, tower: null },
    });
  });

  it("explains unusable decks", () => {
    const ladderOnly = makeAnalysis({ battles: [makeBattle({ mode: "Ladder" })] });
    expect(latestRankedDeck(ladderOnly, matchupOptions)).toEqual({
      message: "No recent ranked battle.",
    });
    expect(latestRankedDeck(makeAnalysis(), matchupOptions)).toEqual({
      message: "Your latest ranked deck has cards this model doesn't support.",
    });
    const noTower = makeAnalysis({ battles: [makeBattle({ player_cards: cards(names) })] });
    expect(latestRankedDeck(noTower, matchupOptions)).toEqual({
      message: "Your latest ranked tower troop isn't supported.",
    });
  });
});
