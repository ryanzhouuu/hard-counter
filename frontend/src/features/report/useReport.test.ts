import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { jsonResponse, makeAnalysis } from "../../test/fixtures";
import { useReport } from "./useReport";

function stubReports() {
  const fetchMock = vi.fn(() => Promise.resolve(jsonResponse(makeAnalysis())));
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

beforeEach(() => {
  window.history.replaceState(null, "", "/");
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("useReport", () => {
  it("loads the tag in the URL with the default window", async () => {
    window.history.replaceState(null, "", "/?tag=abc123");
    const fetchMock = stubReports();

    const { result } = renderHook(() => useReport());

    expect(result.current.loading).toBe(true);
    await waitFor(() => expect(result.current.report?.player.name).toBe("Ryan"));
    expect(fetchMock.mock.calls[0]).toEqual([
      "/api/player-analysis?tag=ABC123&window=10",
      expect.anything(),
    ]);
  });

  it("puts lookups in the URL and reuses cached windows", async () => {
    const fetchMock = stubReports();
    const { result } = renderHook(() => useReport());
    expect(result.current.query).toBeNull();

    act(() => result.current.lookUp("ABC123"));
    expect(window.location.search).toBe("?tag=ABC123&window=10");
    await waitFor(() => expect(result.current.loading).toBe(false));

    act(() => result.current.selectWindow(5));
    expect(result.current.report).not.toBeNull();
    await waitFor(() => expect(result.current.loading).toBe(false));

    act(() => result.current.selectWindow(10));
    expect(result.current.loading).toBe(false);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("keeps an error until a retry succeeds", async () => {
    window.history.replaceState(null, "", "/?tag=ABC123&window=5");
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ detail: "Rate limited." }, 429))
      .mockResolvedValueOnce(jsonResponse(makeAnalysis()));
    vi.stubGlobal("fetch", fetchMock);

    const { result } = renderHook(() => useReport());
    await waitFor(() => expect(result.current.error?.status).toBe(429));

    act(() => result.current.retry());
    await waitFor(() => expect(result.current.report).not.toBeNull());
    expect(result.current.error).toBeNull();
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("follows browser history", async () => {
    stubReports();
    const { result } = renderHook(() => useReport());
    act(() => result.current.lookUp("ABC123"));
    await waitFor(() => expect(result.current.loading).toBe(false));

    act(() => {
      window.history.pushState(null, "", "/");
      window.dispatchEvent(new PopStateEvent("popstate"));
    });

    expect(result.current.query).toBeNull();
  });
});
