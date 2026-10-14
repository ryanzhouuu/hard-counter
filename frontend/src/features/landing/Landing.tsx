import { useState } from "react";

import { clearRecentPlayers, readRecentPlayers } from "../../lib/recentPlayers";
import { queryString } from "../report/query";
import { TagForm } from "./TagForm";
import "./landing.css";

function Landing({ onLookUp }: { onLookUp: (tag: string) => void }) {
  const [recent, setRecent] = useState(readRecentPlayers);
  return (
    <main className="landing">
      <h1 className="landing-title">Hard Counter</h1>
      <p className="landing-tagline">How tough were your recent matches?</p>
      <TagForm size="large" onLookUp={onLookUp} />
      <p className="landing-hint">Your tag is on your Clash Royale profile, under your name.</p>
      {recent.length > 0 && (
        <nav className="landing-recent" aria-label="Recent players">
          <span className="landing-recent-label">Recent</span>
          {recent.map((player) => (
            <a key={player.tag} href={queryString({ tag: player.tag, windowSize: player.window })}>
              {player.name} <span className="num">#{player.tag}</span>
            </a>
          ))}
          <button
            type="button"
            onClick={() => {
              clearRecentPlayers();
              setRecent([]);
            }}
          >
            Clear
          </button>
        </nav>
      )}
      <a className="matchup-link landing-matchup" href="?view=matchup">
        Predict a deck matchup →
      </a>
    </main>
  );
}

export { Landing };
