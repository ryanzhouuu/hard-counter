import { WINDOW_SIZES, type WindowSize } from "./query";

type WindowPickerProps = {
  selected: WindowSize;
  available: number;
  onSelect: (windowSize: WindowSize) => void;
};

function WindowPicker({ selected, available, onSelect }: WindowPickerProps) {
  return (
    <fieldset className="window-picker">
      <legend>Last</legend>
      <div className="segmented">
        {WINDOW_SIZES.map((size) => (
          <button
            key={size}
            type="button"
            aria-pressed={size === selected}
            disabled={size !== selected && size > available}
            onClick={() => onSelect(size)}
          >
            {size} battles
          </button>
        ))}
      </div>
    </fieldset>
  );
}

export { WindowPicker };
