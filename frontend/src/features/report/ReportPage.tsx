import type { AnalysisError, PlayerAnalysis } from "../../api/playerAnalysis";
import { TagForm } from "../landing/TagForm";
import { BattleList } from "./BattleList";
import { ErrorNotice } from "./ErrorNotice";
import type { WindowSize } from "./query";
import { ReportSkeleton } from "./ReportSkeleton";
import { Summary } from "./Summary";
import "./report.css";

type ReportPageProps = {
  tag: string;
  windowSize: WindowSize;
  report: PlayerAnalysis | null;
  loading: boolean;
  error: AnalysisError | null;
  onLookUp: (tag: string) => void;
  onSelectWindow: (windowSize: WindowSize) => void;
  onRetry: () => void;
};

function ReportPage(props: ReportPageProps) {
  const { tag, windowSize, report, loading, error } = props;
  return (
    <div className="report-page">
      <header className="topbar">
        <a className="wordmark" href="./">
          Clash SoS
        </a>
        <TagForm key={tag} initialTag={tag} onLookUp={props.onLookUp} />
      </header>
      {error && <ErrorNotice error={error} onRetry={props.onRetry} />}
      {report ? (
        <main className="report" aria-busy={loading} data-updating={loading || undefined}>
          <Summary report={report} windowSize={windowSize} onSelectWindow={props.onSelectWindow} />
          <BattleList
            key={report.player.tag}
            battles={report.battles}
            windowFilled={report.schedule.status === "available"}
          />
        </main>
      ) : (
        loading && <ReportSkeleton />
      )}
    </div>
  );
}

export { ReportPage };
