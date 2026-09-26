import { TagForm } from "./TagForm";
import "./landing.css";

function Landing({ onLookUp }: { onLookUp: (tag: string) => void }) {
  return (
    <main className="landing">
      <h1 className="landing-title">Clash SoS</h1>
      <p className="landing-tagline">How tough were your recent matches?</p>
      <TagForm size="large" onLookUp={onLookUp} />
    </main>
  );
}

export { Landing };
