import { useEffect } from "react";

function isEditable(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
}

function useShortcuts(): void {
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.metaKey || event.ctrlKey || event.altKey || isEditable(event.target)) return;
      const rows = [...document.querySelectorAll<HTMLButtonElement>(".battle-row")];
      const index = rows.findIndex((row) => row === document.activeElement);
      if (event.key === "/") {
        event.preventDefault();
        document.querySelector<HTMLInputElement>('input[aria-label="Player tag"]')?.focus();
      } else if ((event.key === "j" || event.key === "k") && rows.length > 0) {
        event.preventDefault();
        const step = event.key === "j" ? 1 : -1;
        const next = index === -1 ? 0 : Math.min(Math.max(index + step, 0), rows.length - 1);
        rows[next].focus();
      } else if (event.key === "Escape" && rows[index]?.getAttribute("aria-expanded") === "true") {
        rows[index].click();
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);
}

export { useShortcuts };
