import { type AnalysisError, isRetryable } from "../../api/playerAnalysis";

function ErrorNotice({ error, onRetry }: { error: AnalysisError; onRetry: () => void }) {
  return (
    <div className="error-notice" role="alert">
      <p>{error.message}</p>
      {isRetryable(error) && (
        <button type="button" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}

export { ErrorNotice };
