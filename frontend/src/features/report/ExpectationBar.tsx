import { signedValue } from "../../lib/format";

type ExpectationBarProps = { actual: number; expected: number; total: number };

function ExpectationBar({ actual, expected, total }: ExpectationBarProps) {
  const { tone } = signedValue(actual - expected);
  const low = (Math.min(actual, expected) / total) * 100;
  const high = (Math.max(actual, expected) / total) * 100;
  return (
    <figure className="expectation">
      <div className="expectation-track" aria-hidden="true">
        <span className="expectation-base" style={{ width: `${low}%` }} />
        <span
          className={`expectation-gap gap-${tone}`}
          style={{ left: `${low}%`, width: `${high - low}%` }}
        />
      </div>
      <figcaption className="expectation-legend">
        <span>{tone === "negative" ? "Won" : "Expected"}</span>
        {tone !== "neutral" && (
          <span className={`gap-${tone}`}>
            {tone === "positive" ? "Above expected" : "Below expected"}
          </span>
        )}
      </figcaption>
    </figure>
  );
}

export { ExpectationBar };
