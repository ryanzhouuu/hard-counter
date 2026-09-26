import { afterEach, describe, expect, it, vi } from "vitest";

import { jsonResponse, makeAnalysis } from "../test/fixtures";
import { AnalysisError, fetchPlayerAnalysis, isRetryable } from "./playerAnalysis";

async function failure(request: Promise<unknown>): Promise<AnalysisError> {
  const error = await request.catch((reason: unknown) => reason);
  expect(error).toBeInstanceOf(AnalysisError);
  return error as AnalysisError;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("fetchPlayerAnalysis", () => {
  it("requests the tag and window and returns the report", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(makeAnalysis()));
    vi.stubGlobal("fetch", fetchMock);

    const report = await fetchPlayerAnalysis("ABC123", 10);

    expect(fetchMock).toHaveBeenCalledWith("/api/player-analysis?tag=ABC123&window=10", {
      signal: undefined,
    });
    expect(report.player.name).toBe("Ryan");
  });

  it("uses the API message for failed lookups", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse({ detail: "Player tag not found." }, 404)),
    );

    const error = await failure(fetchPlayerAnalysis("ABC123", 10));

    expect(error).toMatchObject({ message: "Player tag not found.", status: 404 });
    expect(isRetryable(error)).toBe(false);
  });

  it("falls back to a generic message when the detail is not text", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse({ detail: [{ msg: "bad window" }] }, 422)),
    );

    const error = await failure(fetchPlayerAnalysis("ABC123", 10));

    expect(error).toMatchObject({ message: "Something went wrong. Try again.", status: 422 });
  });

  it("reports an unreachable server as retryable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));

    const error = await failure(fetchPlayerAnalysis("ABC123", 10));

    expect(error).toMatchObject({ message: "Could not reach the server. Try again.", status: 0 });
    expect(isRetryable(error)).toBe(true);
  });

  it("treats rate limits and upstream outages as retryable", () => {
    expect(isRetryable(new AnalysisError("", 429))).toBe(true);
    expect(isRetryable(new AnalysisError("", 502))).toBe(true);
    expect(isRetryable(new AnalysisError("", 503))).toBe(true);
  });
});
