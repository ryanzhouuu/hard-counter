import type { PlayerAnalysis, Schedule } from "../../api/playerAnalysis";
import { percent, signedValue } from "../../lib/format";
import { ExpectationBar } from "./ExpectationBar";
import type { WindowSize } from "./query";
import { WindowPicker } from "./WindowPicker";
import "./summary.css";

type SummaryProps = {
  report: PlayerAnalysis;
  windowSize: WindowSize;
  onSelectWindow: (windowSize: WindowSize) => void;
};

function playerMeta({ tag, trophies }: PlayerAnalysis["player"]): string {
  return trophies === null ? tag : `${tag} · ${trophies.toLocaleString("en-US")} trophies`;
}

function shortfallNote(report: PlayerAnalysis): string {
  const count = report.schedule.eligible_count;
  if (report.battles.length === 0) return "No recent battles.";
  if (count === 0) return "No scored battles.";
  return `Only ${count} scored ${count === 1 ? "battle" : "battles"}`;
}

function Headline({ schedule }: { schedule: Schedule }) {
  const { actual_wins, expected_wins, performance_above_expectation, strength_of_schedule } =
    schedule;
  if (
    actual_wins === null ||
    expected_wins === null ||
    performance_above_expectation === null ||
    strength_of_schedule === null
  ) {
    return null;
  }
  const delta = signedValue(performance_above_expectation);
  return (
    <>
      <p className="headline num">
        Won {actual_wins} of {schedule.requested_window}
      </p>
      <p className="subline num">
        Expected {expected_wins.toFixed(1)} ·{" "}
        <span className={`tone-${delta.tone}`}>{delta.text}</span>
      </p>
      <ExpectationBar
        actual={actual_wins}
        expected={expected_wins}
        total={schedule.requested_window}
      />
      <p className="stat num">
        <span>Avg win chance</span> {percent(1 - strength_of_schedule)}
      </p>
    </>
  );
}

function Summary({ report, windowSize, onSelectWindow }: SummaryProps) {
  const { player, schedule } = report;
  return (
    <aside className="summary" aria-label="Summary">
      <h1 className="player-name">{player.name}</h1>
      <p className="player-meta num">{playerMeta(player)}</p>
      {schedule.status === "available" ? (
        <Headline schedule={schedule} />
      ) : (
        <p className="headline-note">{shortfallNote(report)}</p>
      )}
      <WindowPicker
        selected={windowSize}
        available={schedule.eligible_count}
        onSelect={onSelectWindow}
      />
    </aside>
  );
}

export { Summary };
