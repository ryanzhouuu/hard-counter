import { useState } from "react";

import { normalizeTag } from "../../lib/tag";

type TagFormProps = {
  initialTag?: string;
  size?: "large" | "compact";
  onLookUp: (tag: string) => void;
};

function TagForm({ initialTag = "", size = "compact", onLookUp }: TagFormProps) {
  const [value, setValue] = useState(initialTag);
  const [invalid, setInvalid] = useState(false);

  return (
    <form
      className={`tag-form tag-form-${size}`}
      role="search"
      noValidate
      onSubmit={(event) => {
        event.preventDefault();
        const tag = normalizeTag(value);
        setInvalid(tag === null);
        if (tag !== null) onLookUp(tag);
      }}
    >
      <div className="tag-field">
        <span className="tag-hash" aria-hidden="true">
          #
        </span>
        <input
          aria-label="Player tag"
          placeholder="Player tag"
          value={value}
          autoCapitalize="characters"
          autoComplete="off"
          spellCheck={false}
          aria-invalid={invalid}
          onChange={(event) => setValue(event.target.value)}
        />
        <button type="submit">Look up</button>
      </div>
      {invalid && (
        <p className="tag-error" role="alert">
          Enter a valid player tag.
        </p>
      )}
    </form>
  );
}

export { TagForm };
