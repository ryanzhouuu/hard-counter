import { useEffect, useState } from "react";

async function copyPageLink(): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(window.location.href);
    return true;
  } catch {
    const previousFocus = document.activeElement;
    const input = document.createElement("textarea");
    input.value = window.location.href;
    input.readOnly = true;
    input.style.position = "fixed";
    input.style.top = "0";
    input.style.left = "0";
    input.style.opacity = "0";
    input.setAttribute("aria-hidden", "true");
    input.tabIndex = -1;
    document.body.append(input);
    input.select();
    try {
      return document.execCommand("copy");
    } catch {
      return false;
    } finally {
      input.remove();
      if (previousFocus instanceof HTMLElement) previousFocus.focus();
    }
  }
}

function CopyLink() {
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    if (!copied) return;
    const timer = setTimeout(() => setCopied(false), 2000);
    return () => clearTimeout(timer);
  }, [copied]);
  return (
    <button
      type="button"
      className="copy-link"
      onClick={() => {
        void copyPageLink().then(setCopied);
      }}
    >
      {copied ? "Copied" : "Copy link"}
    </button>
  );
}

export { CopyLink };
