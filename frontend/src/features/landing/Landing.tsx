import { TagForm } from "./TagForm";
import "./landing.css";

function Landing({ onLookUp }: { onLookUp: (tag: string) => void }) {
  return (
    <main className="landing">
      <h1 className="landing-title">Hard Counter</h1>
      <p className="landing-tagline">How tough were your recent matches?</p>
      <TagForm size="large" onLookUp={onLookUp} />
      <p className="landing-hint">Your tag is on your Clash Royale profile, under your name.</p>
    </main>
  );
}

export { Landing };
