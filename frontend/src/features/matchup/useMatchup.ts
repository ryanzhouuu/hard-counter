import { useEffect, useRef, useState } from "react";

import {
  fetchMatchupOptions,
  predictMatchup,
  type MatchupDeck,
  type MatchupOptions,
  type MatchupRequest,
  type MatchupResult,
} from "../../api/deckMatchup";
import { AnalysisError, FALLBACK_MESSAGE } from "../../api/playerAnalysis";

function asError(error: unknown): AnalysisError {
  return error instanceof AnalysisError ? error : new AnalysisError(FALLBACK_MESSAGE, 0);
}

function useMatchup() {
  const [options, setOptions] = useState<MatchupOptions | null>(null);
  const [optionsError, setOptionsError] = useState<AnalysisError | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [decks, setDecks] = useState<MatchupRequest>({
    side_a: { cards: [], tower: null },
    side_b: { cards: [], tower: null },
  });
  const [result, setResult] = useState<MatchupResult | null>(null);
  const [error, setError] = useState<AnalysisError | null>(null);
  const [predicting, setPredicting] = useState(false);
  const pending = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    fetchMatchupOptions(controller.signal).then(
      (value) => {
        if (!controller.signal.aborted) setOptions(value);
      },
      (failure: unknown) => {
        if (!controller.signal.aborted) setOptionsError(asError(failure));
      },
    );
    return () => controller.abort();
  }, [attempt]);

  useEffect(() => () => pending.current?.abort(), []);

  const ready =
    options !== null &&
    [decks.side_a, decks.side_b].every(
      (deck) =>
        deck.cards.length === 8 &&
        (options.model.input_scope === "deck_only" || deck.tower !== null),
    );

  function clearResult() {
    pending.current?.abort();
    setResult(null);
    setError(null);
    setPredicting(false);
  }

  function changeDeck(side: keyof MatchupRequest, deck: MatchupDeck) {
    clearResult();
    setDecks((current) => ({ ...current, [side]: deck }));
  }

  function swap() {
    clearResult();
    setDecks((current) => ({ side_a: current.side_b, side_b: current.side_a }));
  }

  async function predict() {
    if (!ready || predicting) return;
    const controller = new AbortController();
    pending.current = controller;
    setPredicting(true);
    setResult(null);
    setError(null);
    try {
      const value = await predictMatchup(decks, controller.signal);
      if (!controller.signal.aborted) setResult(value);
    } catch (failure) {
      if (!controller.signal.aborted) setError(asError(failure));
    } finally {
      if (!controller.signal.aborted) setPredicting(false);
    }
  }

  function retryOptions() {
    setOptionsError(null);
    setAttempt((current) => current + 1);
  }

  return {
    options,
    optionsError,
    decks,
    ready,
    predicting,
    result,
    error,
    changeDeck,
    swap,
    predict,
    retryOptions,
  };
}

export { useMatchup };
