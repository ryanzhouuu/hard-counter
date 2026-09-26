type Tone = "positive" | "negative" | "neutral";

const SKIP_LABELS: Record<string, string> = {
  draw: "Draw",
  team_battle: "Team battle",
  unknown_card: "New card not yet supported",
  model_coverage: "Card not supported by this model",
  incomplete_deck: "Incomplete deck",
  repeated_card: "Unusual deck",
  incomplete_battle: "Incomplete battle data",
  player_not_in_battle: "Player not found in battle",
};

function signedValue(value: number): { text: string; tone: Tone } {
  const rounded = Math.round(value * 10) / 10;
  if (rounded > 0) return { text: `+${rounded.toFixed(1)}`, tone: "positive" };
  if (rounded < 0) return { text: `−${Math.abs(rounded).toFixed(1)}`, tone: "negative" };
  return { text: "0.0", tone: "neutral" };
}

function percent(probability: number): string {
  return `${Math.round(probability * 100)}%`;
}

function relativeTime(timestamp: string | null, now: Date = new Date()): string {
  if (timestamp === null) return "";
  const minutes = Math.floor((now.getTime() - Date.parse(timestamp)) / 60_000);
  if (minutes < 1) return "now";
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  return hours < 24 ? `${hours}h` : `${Math.floor(hours / 24)}d`;
}

function skipLabel(reason: string | null): string {
  return (reason !== null && SKIP_LABELS[reason]) || "Not scored";
}

function modeLabel(mode: string): string | null {
  if (mode.startsWith("Ranked1v1")) return "Ranked";
  return mode === "Ladder" ? "Ladder" : null;
}

export { modeLabel, percent, relativeTime, signedValue, skipLabel };
export type { Tone };
