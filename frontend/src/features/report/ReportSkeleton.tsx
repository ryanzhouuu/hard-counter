const ROW_COUNT = 6;

function ReportSkeleton() {
  return (
    <main className="report" aria-busy="true" aria-label="Loading report">
      <div className="summary">
        <span className="skeleton skeleton-name" />
        <span className="skeleton skeleton-headline" />
        <span className="skeleton skeleton-line" />
      </div>
      <ul className="battle-list skeleton-list">
        {Array.from({ length: ROW_COUNT }, (_, index) => (
          <li key={index} className="skeleton skeleton-row" />
        ))}
      </ul>
    </main>
  );
}

export { ReportSkeleton };
