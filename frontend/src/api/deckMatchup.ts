import { AnalysisError } from "./playerAnalysis";

type MatchupModel = {
  model_version: string;
  dataset_version: string;
  catalog_version: string;
  training_era_id: string;
  input_scope: "deck_only" | "deck_and_tower";
};
type CardChoice = {
  identity: string;
  name: string;
  form: string;
  icon_url: string | null;
  elixir: number | null;
};
type TowerChoice = { identity: string; name: string; icon_url: string | null };
type MatchupOptions = { model: MatchupModel; cards: CardChoice[]; towers: TowerChoice[] };
type MatchupDeck = { cards: string[]; tower: string | null };
type MatchupRequest = { side_a: MatchupDeck; side_b: MatchupDeck };
type MatchupResult = {
  model: MatchupModel;
  side_a_win_probability: number;
  side_b_win_probability: number;
};

async function request<T>(path: string, init: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, init);
  } catch (error) {
    if (init.signal?.aborted) throw error;
    throw new AnalysisError("Could not reach the server. Try again.", 0);
  }
  if (!response.ok) {
    const body: unknown = await response.json().catch(() => null);
    const detail =
      typeof body === "object" &&
      body !== null &&
      "detail" in body &&
      typeof body.detail === "string"
        ? body.detail
        : "Check both decks and try again.";
    throw new AnalysisError(detail, response.status);
  }
  return (await response.json()) as T;
}

function fetchMatchupOptions(signal?: AbortSignal): Promise<MatchupOptions> {
  return request("/api/matchup-options", { signal });
}

function predictMatchup(body: MatchupRequest, signal?: AbortSignal): Promise<MatchupResult> {
  return request("/api/matchup-prediction", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
}

export { fetchMatchupOptions, predictMatchup };
export type { CardChoice, MatchupDeck, MatchupOptions, MatchupRequest, MatchupResult, TowerChoice };
