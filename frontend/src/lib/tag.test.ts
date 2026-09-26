import { describe, expect, it } from "vitest";

import { normalizeTag } from "./tag";

describe("normalizeTag", () => {
  it.each([
    ["#abc123", "ABC123"],
    [" abc123 ", "ABC123"],
    ["ABC", "ABC"],
  ])("accepts %j", (raw, expected) => {
    expect(normalizeTag(raw)).toBe(expected);
  });

  it.each(["", "#", "ab", "abc-12", "##ABC", "A".repeat(16)])("rejects %j", (raw) => {
    expect(normalizeTag(raw)).toBeNull();
  });
});
