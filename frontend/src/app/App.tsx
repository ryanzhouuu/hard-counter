import { Landing } from "../features/landing/Landing";
import { MatchupPage } from "../features/matchup/MatchupPage";
import { ReportPage } from "../features/report/ReportPage";
import { useReport } from "../features/report/useReport";
import "./tokens.css";

const DISCLAIMER = "Unofficial fan project. Not affiliated with or endorsed by Supercell.";

function App() {
  const matchup = new URLSearchParams(window.location.search).get("view") === "matchup";
  return (
    <div className="app">
      {matchup ? <MatchupPage /> : <PlayerView />}
      <footer className="disclaimer">{DISCLAIMER}</footer>
    </div>
  );
}

function PlayerView() {
  const view = useReport();
  return (
    <>
      {view.query === null ? (
        <Landing onLookUp={view.lookUp} />
      ) : (
        <ReportPage
          tag={view.query.tag}
          windowSize={view.query.windowSize}
          report={view.report}
          loading={view.loading}
          error={view.error}
          onLookUp={view.lookUp}
          onSelectWindow={view.selectWindow}
          onRetry={view.retry}
        />
      )}
    </>
  );
}

export { App };
