import { useState } from "react";

import { OverviewPage } from "../features/overview/OverviewPage";
import "./app.css";
import "./overview.css";
import "./responsive.css";

const navItems = ["Overview", "Matchups", "Schedule", "Models"] as const;
type NavItem = (typeof navItems)[number];

function MarkIcon() {
  return (
    <span className="brand-mark" aria-hidden="true">
      ⌁
    </span>
  );
}

function App() {
  const [activeView, setActiveView] = useState<NavItem>("Overview");

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <MarkIcon />
          <div>
            <p className="eyebrow">CLASH ROYALE</p>
            <p className="brand-name">Clash SoS</p>
          </div>
        </div>

        <nav className="primary-nav" aria-label="Primary navigation">
          <p className="nav-label">Workspace</p>
          {navItems.map((item) => (
            <button
              className={`nav-item${activeView === item ? " active" : ""}`}
              key={item}
              type="button"
              aria-current={activeView === item ? "page" : undefined}
              onClick={() => setActiveView(item)}
            >
              <span className="nav-icon" aria-hidden="true">
                {item === "Overview" ? "◈" : "·"}
              </span>
              {item}
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="sync-badge">
            <span className="status-dot" />
            Local workspace
          </div>
          <p>Model artifacts stay on this machine until you choose to publish them.</p>
        </div>
      </aside>

      <section className="content-area">
        {activeView === "Overview" ? (
          <OverviewPage />
        ) : (
          <section className="empty-view" aria-live="polite">
            <p className="eyebrow accent">COMING NEXT</p>
            <h1>{activeView}</h1>
            <p>The analysis surface is scaffolded and waiting for its first data pipeline.</p>
            <button className="text-button" type="button" onClick={() => setActiveView("Overview")}>
              Return to overview <span aria-hidden="true">↗</span>
            </button>
          </section>
        )}
      </section>
    </main>
  );
}

export { App };
