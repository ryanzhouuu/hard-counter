type Card = {
  name: string;
  icon_url: string | null;
};

type Outcome = "win" | "loss" | "draw" | "unknown";

type Tower = Card & {
  identity: string | null;
  level: number | null;
};

type Battle = {
  timestamp: string | null;
  mode: string;
  opponent_name: string;
  outcome: Outcome;
  player_cards: Card[];
  opponent_cards: Card[];
  player_tower: Tower | null;
  opponent_tower: Tower | null;
  skip_reason: string | null;
  win_probability: number | null;
  in_window: boolean;
};

type Schedule = {
  status: "available" | "insufficient_data";
  requested_window: number;
  eligible_count: number;
  excluded_count: number;
  strength_of_schedule: number | null;
  expected_wins: number | null;
  actual_wins: number | null;
  performance_above_expectation: number | null;
};

type PlayerAnalysis = {
  player: { tag: string; name: string; trophies: number | null };
  model: {
    model_version: string;
    dataset_version: string;
    catalog_version: string;
    training_era_id: string;
    input_scope: "deck_only" | "deck_and_tower";
  };
  schedule: Schedule;
  battles: Battle[];
};

const FALLBACK_MESSAGE = "Something went wrong. Try again.";
const UNREACHABLE_MESSAGE = "Could not reach the server. Try again.";
const RETRYABLE_STATUSES = new Set([0, 429, 502, 503]);

class AnalysisError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "AnalysisError";
    this.status = status;
  }
}

function isRetryable(error: AnalysisError): boolean {
  return RETRYABLE_STATUSES.has(error.status);
}

async function detailOf(response: Response): Promise<string> {
  const body: unknown = await response.json().catch(() => null);
  if (typeof body === "object" && body !== null && "detail" in body) {
    if (typeof body.detail === "string") return body.detail;
  }
  return FALLBACK_MESSAGE;
}

async function fetchPlayerAnalysis(
  tag: string,
  windowSize: number,
  signal?: AbortSignal,
): Promise<PlayerAnalysis> {
  const query = new URLSearchParams({ tag, window: String(windowSize) });
  let response: Response;
  try {
    response = await fetch(`/api/player-analysis?${query}`, { signal });
  } catch (error) {
    if (signal?.aborted) throw error;
    throw new AnalysisError(UNREACHABLE_MESSAGE, 0);
  }
  if (!response.ok) throw new AnalysisError(await detailOf(response), response.status);
  return (await response.json()) as PlayerAnalysis;
}

export { AnalysisError, FALLBACK_MESSAGE, fetchPlayerAnalysis, isRetryable };
export type { Battle, Card, Outcome, PlayerAnalysis, Schedule, Tower };
