import { readRecentPlayers } from "../../lib/recentPlayers";
import { TagForm } from "../landing/TagForm";
import { queryString } from "../report/query";
import "./topbar.css";

type TopBarProps = {
  page: "players" | "matchup";
  tag?: string;
  onLookUp?: (tag: string) => void;
};

function playersHref(): string {
  const [latest] = readRecentPlayers();
  return latest ? queryString({ tag: latest.tag, windowSize: latest.window }) : "./";
}

function TopBar({ page, tag, onLookUp }: TopBarProps) {
  return (
    <header className="topbar">
      <a className="wordmark" href="./">
        Hard Counter
      </a>
      <nav className="topbar-tabs" aria-label="Pages">
        <a href={playersHref()} aria-current={page === "players" ? "page" : undefined}>
          Players
        </a>
        <a href="?view=matchup" aria-current={page === "matchup" ? "page" : undefined}>
          Matchup
        </a>
      </nav>
      {onLookUp && <TagForm key={tag} initialTag={tag} onLookUp={onLookUp} />}
    </header>
  );
}

export { TopBar };
