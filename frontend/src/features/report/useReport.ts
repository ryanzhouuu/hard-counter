import { useEffect, useState } from "react";

import {
  AnalysisError,
  FALLBACK_MESSAGE,
  fetchPlayerAnalysis,
  type PlayerAnalysis,
} from "../../api/playerAnalysis";
import {
  DEFAULT_WINDOW,
  queryKey,
  queryString,
  readQuery,
  type ReportQuery,
  type WindowSize,
} from "./query";

type Outcome = { report: PlayerAnalysis } | { error: AnalysisError };
type Outcomes = ReadonlyMap<string, Outcome>;

type ReportView = {
  query: ReportQuery | null;
  report: PlayerAnalysis | null;
  loading: boolean;
  error: AnalysisError | null;
  lookUp: (tag: string) => void;
  selectWindow: (windowSize: WindowSize) => void;
  retry: () => void;
};

function latestReport(outcomes: Outcomes, tag: string): PlayerAnalysis | null {
  let latest: PlayerAnalysis | null = null;
  for (const [key, outcome] of outcomes) {
    if (key.startsWith(`${tag}:`) && "report" in outcome) latest = outcome.report;
  }
  return latest;
}

function withoutError(outcomes: Outcomes, key: string): Outcomes {
  const outcome = outcomes.get(key);
  if (outcome === undefined || "report" in outcome) return outcomes;
  const next = new Map(outcomes);
  next.delete(key);
  return next;
}

function asAnalysisError(error: unknown): AnalysisError {
  return error instanceof AnalysisError ? error : new AnalysisError(FALLBACK_MESSAGE, 0);
}

function useReport(): ReportView {
  const [query, setQuery] = useState(() => readQuery(window.location.search));
  const [outcomes, setOutcomes] = useState<Outcomes>(() => new Map());

  useEffect(() => {
    const syncFromUrl = () => setQuery(readQuery(window.location.search));
    window.addEventListener("popstate", syncFromUrl);
    return () => window.removeEventListener("popstate", syncFromUrl);
  }, []);

  useEffect(() => {
    if (query === null || outcomes.has(queryKey(query))) return;
    const key = queryKey(query);
    const controller = new AbortController();
    const settle = (outcome: Outcome) =>
      setOutcomes((current) => new Map(current).set(key, outcome));
    fetchPlayerAnalysis(query.tag, query.windowSize, controller.signal).then(
      (report) => settle({ report }),
      (error: unknown) => {
        if (!controller.signal.aborted) settle({ error: asAnalysisError(error) });
      },
    );
    return () => controller.abort();
  }, [query, outcomes]);

  function navigate(next: ReportQuery, history: "push" | "replace" = "push") {
    const url = queryString(next);
    if (history === "push") window.history.pushState(null, "", url);
    else window.history.replaceState(null, "", url);
    setOutcomes((current) => withoutError(current, queryKey(next)));
    setQuery(next);
  }

  const outcome = query === null ? undefined : outcomes.get(queryKey(query));
  const currentReport = outcome && "report" in outcome ? outcome.report : null;
  return {
    query,
    report: currentReport ?? (query === null ? null : latestReport(outcomes, query.tag)),
    loading: query !== null && outcome === undefined,
    error: outcome && "error" in outcome ? outcome.error : null,
    lookUp: (tag) => navigate({ tag, windowSize: query?.windowSize ?? DEFAULT_WINDOW }),
    selectWindow: (windowSize) => {
      if (query !== null) navigate({ ...query, windowSize });
    },
    retry: () => {
      if (query !== null) navigate(query, "replace");
    },
  };
}

export { useReport };
