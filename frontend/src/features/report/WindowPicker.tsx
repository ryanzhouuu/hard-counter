import { useId } from "react";

import { WINDOW_SIZES, type WindowSize } from "./query";

type WindowPickerProps = {
  selected: WindowSize;
  available: number;
  explainLimit?: boolean;
  onSelect: (windowSize: WindowSize) => void;
};

function WindowPicker({ selected, available, explainLimit = false, onSelect }: WindowPickerProps) {
  const noteId = useId();
  const isDisabled = (size: WindowSize) => size !== selected && size > available;
  const showNote = explainLimit && WINDOW_SIZES.some(isDisabled);
  return (
    <fieldset className="window-picker" aria-describedby={showNote ? noteId : undefined}>
      <legend className="visually-hidden">Battles counted</legend>
      <div className="window-row">
        <span aria-hidden="true">Last</span>
        <div className="segmented">
          {WINDOW_SIZES.map((size) => (
            <button
              key={size}
              type="button"
              aria-label={`${size} battles`}
              aria-pressed={size === selected}
              disabled={isDisabled(size)}
              onClick={() => onSelect(size)}
            >
              {size}
            </button>
          ))}
        </div>
        <span aria-hidden="true">battles</span>
      </div>
      {showNote && (
        <p id={noteId} className="window-note">
          Only {available} scored {available === 1 ? "battle" : "battles"}
        </p>
      )}
    </fieldset>
  );
}

export { WindowPicker };
