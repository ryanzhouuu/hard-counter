import type { PlayerAnalysis } from "../api/playerAnalysis";
import { isWindowSize, type ReportQuery, type WindowSize } from "../features/report/query";

type RecentPlayer = { tag: string; name: string; window: WindowSize };

const STORAGE_KEY = "hard-counter:recent-players";
const LIMIT = 5;

function isRecentPlayer(value: unknown): value is RecentPlayer {
  if (typeof value !== "object" || value === null) return false;
  const entry = value as Record<string, unknown>;
  return (
    typeof entry.tag === "string" &&
    typeof entry.name === "string" &&
    typeof entry.window === "number" &&
    isWindowSize(entry.window)
  );
}

function readRecentPlayers(): RecentPlayer[] {
  try {
    const parsed: unknown = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]");
    return Array.isArray(parsed) ? parsed.filter(isRecentPlayer).slice(0, LIMIT) : [];
  } catch {
    return [];
  }
}

function saveRecentPlayer(player: RecentPlayer): void {
  const next = [player, ...readRecentPlayers().filter((entry) => entry.tag !== player.tag)];
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next.slice(0, LIMIT)));
  } catch {
    // Recents are a convenience; storage may be unavailable.
  }
}

function clearRecentPlayers(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // Recents are a convenience; storage may be unavailable.
  }
}

function recentFromReport(
  query: ReportQuery | null,
  report: PlayerAnalysis | null,
  loading: boolean,
): RecentPlayer | null {
  if (query === null || report === null || loading) return null;
  if (report.player.tag !== `#${query.tag}`) return null;
  return { tag: query.tag, name: report.player.name, window: query.windowSize };
}

export { clearRecentPlayers, readRecentPlayers, recentFromReport, saveRecentPlayer };
export type { RecentPlayer };
