import { useEffect, useState } from "react";

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
        navigator.clipboard.writeText(window.location.href).then(
          () => setCopied(true),
          () => undefined,
        );
      }}
    >
      {copied ? "Copied" : "Copy link"}
    </button>
  );
}

export { CopyLink };
