import { Landing } from "../features/landing/Landing";
import { ReportPage } from "../features/report/ReportPage";
import { useReport } from "../features/report/useReport";
import "./tokens.css";

const DISCLAIMER = "Unofficial fan project. Not affiliated with or endorsed by Supercell.";

function App() {
  const view = useReport();
  return (
    <div className="app">
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
      <footer className="disclaimer">{DISCLAIMER}</footer>
    </div>
  );
}

export { App };
