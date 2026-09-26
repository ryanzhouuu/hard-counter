const TAG_PATTERN = /^[A-Z0-9]{3,15}$/;

function normalizeTag(raw: string): string | null {
  const tag = raw.trim().toUpperCase().replace(/^#/, "");
  return TAG_PATTERN.test(tag) ? tag : null;
}

export { normalizeTag };
