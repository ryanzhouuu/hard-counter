import { describe, expect, it } from "vitest";

import { queryString, readQuery } from "./query";

describe("report query", () => {
  it("reads a normalized tag and window", () => {
    expect(readQuery("?tag=%23abc123&window=25")).toEqual({ tag: "ABC123", windowSize: 25 });
  });

  it("defaults unsupported windows to 10", () => {
    expect(readQuery("?tag=ABC123&window=7")).toEqual({ tag: "ABC123", windowSize: 10 });
    expect(readQuery("?tag=ABC123")).toEqual({ tag: "ABC123", windowSize: 10 });
  });

  it("ignores missing or invalid tags", () => {
    expect(readQuery("")).toBeNull();
    expect(readQuery("?tag=!!")).toBeNull();
  });

  it("writes a shareable query string", () => {
    expect(queryString({ tag: "ABC123", windowSize: 5 })).toBe("?tag=ABC123&window=5");
  });
});
