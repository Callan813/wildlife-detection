import { useEffect, useState } from "react";
import { api, pct, type Stats } from "./api";

// Progress, how often the models agree with reviewers, and the "new leopard" threshold
// recalibrated from review decisions.
export function StatsView() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.stats().then(setStats, (e: Error) => setError(e.message));
  }, []);

  if (error) return <p className="error">Couldn't load stats: {error}</p>;
  if (!stats) return <p className="empty">Loading…</p>;
  const { species: s, reid: r } = stats;
  const cal = r.threshold.recalibrated;

  return (
    <section className="stats">
      <div className="card">
        <h2>Species</h2>
        <Progress done={s.reviewed} total={s.total} />
        <dl>
          <dt>Model accuracy (random audit, {s.audit_reviewed} reviewed)</dt>
          <dd>{pct(s.model_accuracy)}</dd>
          <dt>Model right on its hardest cases</dt>
          <dd>{pct(s.model_agrees_on_hardest)}</dd>
          <dt>Reviewers agree with dataset labels</dt>
          <dd>{pct(s.reviewer_agrees_with_dataset)}</dd>
        </dl>
      </div>

      <div className="card">
        <h2>Leopards</h2>
        <Progress done={r.reviewed} total={r.total} />
        <dl>
          <dt>Audit items reviewed</dt>
          <dd>{r.audit_reviewed}</dd>
          <dt>Matches that were the model's #1 (audit)</dt>
          <dd>{pct(r.top1_confirmed)}</dd>
          <dt>"New leopard" suggestion right (audit)</dt>
          <dd>{pct(r.new_suggestion_agrees)}</dd>
          <dt>Reviewers agree with dataset</dt>
          <dd>{pct(r.reviewer_agrees_with_dataset)}</dd>
          <dt>Marked unsure</dt>
          <dd>{r.unsure}</dd>
        </dl>
      </div>

      <div className="card wide">
        <h2>"New leopard" threshold</h2>
        <p>
          Started at <strong>{r.threshold.start.toFixed(2)}</strong>, picked on validation leopards. Thresholds
          don't carry over well between groups of leopards, so reviewer decisions are used to re-tune it.
        </p>
        {cal ? (
          <p>
            From {cal.matches} matches and {cal.news} new leopards in the audit sample, the threshold that best
            agrees with reviewers is <strong>{cal.threshold.toFixed(2)}</strong> (balanced accuracy{" "}
            {pct(cal.balancedAccuracy)}).
          </p>
        ) : (
          <p className="hint">
            Needs at least 10 "match" and 10 "new" decisions on audit items before it can be re-tuned.
          </p>
        )}
      </div>
      <p className="footnote">
        The queue shows the model's least certain items first, and slips in a random audit item every fifth turn.
        Accuracy and the threshold come from audit items only: the uncertain ones are the model's hardest cases by
        design, so they'd make it look worse than it is and say nothing about where the threshold belongs. "Agree
        with dataset" uses labels kept in the export for this demo; a real survey wouldn't have them.
      </p>
    </section>
  );
}

function Progress({ done, total }: { done: number; total: number }) {
  return (
    <div className="progress">
      <div className="progress-bar" aria-hidden>
        <span style={{ width: `${(done / total) * 100}%` }} />
      </div>
      <span>
        {done.toLocaleString()} of {total.toLocaleString()} reviewed
      </span>
    </div>
  );
}
