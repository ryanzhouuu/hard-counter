type Metric = {
  label: string;
  value: string;
  detail: string;
  status: string;
};

const metrics: Metric[] = [
  {
    label: "Matchup probability",
    value: "—",
    detail: "No model loaded",
    status: "Awaiting baseline",
  },
  {
    label: "Rolling strength",
    value: "—",
    detail: "No battle history",
    status: "Awaiting ingestion",
  },
  { label: "Expected wins", value: "—", detail: "No sample window", status: "Needs data" },
  { label: "Above expectation", value: "—", detail: "No comparison yet", status: "Needs a model" },
];

function MetricCard({ metric }: { metric: Metric }) {
  return (
    <article className="metric-card">
      <div className="metric-heading">
        <p>{metric.label}</p>
        <span className="metric-status">{metric.status}</span>
      </div>
      <strong>{metric.value}</strong>
      <p className="metric-detail">{metric.detail}</p>
    </article>
  );
}

function OverviewPage() {
  return (
    <section className="overview" aria-labelledby="overview-title">
      <header className="page-header">
        <div>
          <p className="eyebrow accent">ANALYSIS DESK / 01</p>
          <h1 id="overview-title">
            Read the matchup
            <br />
            <em>before it starts.</em>
          </h1>
        </div>
        <div className="header-meta">
          <span className="live-indicator">
            <span className="status-dot" />
            Local only
          </span>
          <p>
            Workspace ready
            <br />
            <span>Data pipeline not connected</span>
          </p>
        </div>
      </header>

      <div className="analysis-callout">
        <div className="callout-mark" aria-hidden="true">
          ◎
        </div>
        <div>
          <p className="eyebrow">FIRST RUN</p>
          <h2>Your analysis desk is ready for data.</h2>
          <p>
            Connect a battle history or load a versioned model to turn these placeholders into
            signal.
          </p>
        </div>
        <span className="callout-line" aria-hidden="true" />
        <span className="callout-index">
          01 <span>/ 04</span>
        </span>
      </div>

      <div className="metric-grid">
        {metrics.map((metric) => (
          <MetricCard key={metric.label} metric={metric} />
        ))}
      </div>

      <div className="lower-grid">
        <section className="panel timeline-panel" aria-labelledby="timeline-title">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">SIGNAL LOG</p>
              <h2 id="timeline-title">What happens next</h2>
            </div>
            <span className="panel-count">03 steps</span>
          </div>
          <ol className="signal-list">
            <li>
              <span>01</span>
              <div>
                <strong>Bring in battle history</strong>
                <p>Preserve raw payloads before they disappear from the limited log.</p>
              </div>
            </li>
            <li>
              <span>02</span>
              <div>
                <strong>Prepare a clean sample</strong>
                <p>Validate modes, decks, duplicates, and timestamps.</p>
              </div>
            </li>
            <li>
              <span>03</span>
              <div>
                <strong>Evaluate a baseline</strong>
                <p>Measure calibration before trusting a win probability.</p>
              </div>
            </li>
          </ol>
        </section>

        <section className="panel readiness-panel" aria-labelledby="readiness-title">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">SYSTEM READINESS</p>
              <h2 id="readiness-title">Local first</h2>
            </div>
            <span className="readiness-icon">✓</span>
          </div>
          <p className="readiness-copy">
            The scaffold is pointed at a private workspace. Nothing leaves this machine while the
            first baseline is being shaped.
          </p>
          <div className="readiness-bar">
            <span />
          </div>
          <div className="readiness-foot">
            <span>Repository</span>
            <strong>Ready</strong>
          </div>
          <div className="readiness-foot">
            <span>Model registry</span>
            <strong className="muted">Empty</strong>
          </div>
        </section>
      </div>
    </section>
  );
}

export { OverviewPage };
