import { percent } from "../../lib/format";
import { TopBar } from "../nav/TopBar";
import { ErrorNotice } from "../report/ErrorNotice";
import { DEFAULT_WINDOW, queryString } from "../report/query";
import { DeckBuilder } from "./DeckBuilder";
import { useMatchup } from "./useMatchup";
import "./matchup.css";

function MatchupPage() {
  const view = useMatchup();
  const { options, result } = view;
  const model = result?.model ?? options?.model;
  return (
    <div className="matchup-page">
      <TopBar
        page="matchup"
        onLookUp={(tag) => window.location.assign(queryString({ tag, windowSize: DEFAULT_WINDOW }))}
      />
      <main className="matchup-main">
        <div className="matchup-intro">
          <p className="matchup-eyebrow">Head to head</p>
          <h1>Deck matchup</h1>
          <p>Build two decks. See which side the model favors.</p>
        </div>
        {view.optionsError ? (
          <ErrorNotice error={view.optionsError} onRetry={view.retryOptions} />
        ) : options === null ? (
          <p role="status">Loading cards and model…</p>
        ) : (
          <form
            onSubmit={(event) => {
              event.preventDefault();
              void view.predict();
            }}
          >
            <div className="matchup-builders">
              <DeckBuilder
                label="Deck A"
                cards={options.cards}
                towers={options.towers}
                deck={view.decks.side_a}
                onChange={(deck) => view.changeDeck("side_a", deck)}
              />
              <DeckBuilder
                label="Deck B"
                cards={options.cards}
                towers={options.towers}
                deck={view.decks.side_b}
                onChange={(deck) => view.changeDeck("side_b", deck)}
              />
            </div>
            <div className="matchup-actions">
              <p>
                {view.ready
                  ? "Both decks are ready."
                  : `Select 8 cards${options.towers.length ? " and a tower troop" : ""} for each deck.`}
              </p>
              <button
                className="predict-button"
                type="submit"
                disabled={!view.ready || view.predicting}
              >
                {view.predicting ? "Predicting…" : "Predict matchup"}
              </button>
            </div>
          </form>
        )}
        {view.error && <ErrorNotice error={view.error} onRetry={() => void view.predict()} />}
        <div aria-live="polite" aria-atomic="true" aria-busy={view.predicting}>
          {view.predicting && <p role="status">Calculating win chances…</p>}
          {result && (
            <section className="matchup-result" aria-label="Matchup prediction">
              <h2>
                {percent(result.side_a_win_probability) === "50%"
                  ? "Even matchup"
                  : `${result.side_a_win_probability > 0.5 ? "Deck A" : "Deck B"} is favored`}
              </h2>
              <dl className="matchup-probabilities">
                <div>
                  <dt>Deck A win chance</dt>
                  <dd className="num">{percent(result.side_a_win_probability)}</dd>
                </div>
                <div>
                  <dt>Deck B win chance</dt>
                  <dd className="num">{percent(result.side_b_win_probability)}</dd>
                </div>
              </dl>
              <div
                className="matchup-bar"
                role="meter"
                aria-label="Deck A win chance"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={result.side_a_win_probability * 100}
              >
                <span style={{ width: `${result.side_a_win_probability * 100}%` }} />
              </div>
              <p>Estimated win chances assuming equal player skill.</p>
            </section>
          )}
        </div>
        {model && (
          <p className="matchup-provenance">
            Equal-skill estimate ·{" "}
            {model.input_scope === "deck_and_tower" ? "Cards + towers" : "Cards only"}
            <br />
            Model: {model.model_version}
            <br />
            Training era: {model.training_era_id}
          </p>
        )}
      </main>
    </div>
  );
}

export { MatchupPage };
