import { normalizeTag } from "../../lib/tag";

const WINDOW_SIZES = [5, 10, 25] as const;
type WindowSize = (typeof WINDOW_SIZES)[number];
const DEFAULT_WINDOW: WindowSize = 10;

type ReportQuery = { tag: string; windowSize: WindowSize };

function isWindowSize(value: number): value is WindowSize {
  return (WINDOW_SIZES as readonly number[]).includes(value);
}

function readQuery(search: string): ReportQuery | null {
  const params = new URLSearchParams(search);
  const tag = normalizeTag(params.get("tag") ?? "");
  if (tag === null) return null;
  const windowSize = Number(params.get("window"));
  return { tag, windowSize: isWindowSize(windowSize) ? windowSize : DEFAULT_WINDOW };
}

function queryString(query: ReportQuery): string {
  return `?${new URLSearchParams({ tag: query.tag, window: String(query.windowSize) })}`;
}

function queryKey(query: ReportQuery): string {
  return `${query.tag}:${query.windowSize}`;
}

export { DEFAULT_WINDOW, WINDOW_SIZES, queryKey, queryString, readQuery };
export type { ReportQuery, WindowSize };
