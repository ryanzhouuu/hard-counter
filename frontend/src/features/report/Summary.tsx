import type { Battle, PlayerAnalysis, Schedule } from "../../api/playerAnalysis";
import { percent, signedValue } from "../../lib/format";
import { UPSET_DEFINITION, upsetOf } from "../../lib/upset";
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

function upsetCounts(battles: Battle[]): { won: number; lost: number } {
  const upsets = battles.filter((battle) => battle.in_window).map(upsetOf);
  return {
    won: upsets.filter((upset) => upset === "win").length,
    lost: upsets.filter((upset) => upset === "loss").length,
  };
}

function Headline({ schedule, battles }: { schedule: Schedule; battles: Battle[] }) {
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
  const upsets = upsetCounts(battles);
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
      <div className="stats">
        <p className="stat num">
          <span>Avg win chance</span> {percent(1 - strength_of_schedule)}
        </p>
        <p className="stat num" title={UPSET_DEFINITION}>
          <span>Upsets</span> {upsets.won} won · {upsets.lost} lost
        </p>
      </div>
    </>
  );
}

function Summary({ report, windowSize, onSelectWindow }: SummaryProps) {
  const { player, schedule, battles } = report;
  const available = schedule.status === "available";
  return (
    <aside className="summary" aria-label="Summary">
      <h1 className="player-name">{player.name}</h1>
      <p className="player-meta num">{playerMeta(player)}</p>
      {available ? (
        <Headline schedule={schedule} battles={battles} />
      ) : (
        <p className="headline-note">{shortfallNote(report)}</p>
      )}
      <WindowPicker
        selected={windowSize}
        available={schedule.eligible_count}
        explainLimit={available}
        onSelect={onSelectWindow}
      />
    </aside>
  );
}

export { Summary };
