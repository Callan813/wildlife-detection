import { useCallback, useEffect, useState } from "react";
import { api, shortId, type ReidItem, type ReidSummary } from "./api";
import { useQueue } from "./useQueue";

// A new leopard photo next to the model's shortlist of catalogued leopards (same flank).
// Keys: 1-5 same leopard as that candidate, N new leopard, U unsure.
export function ReidReview({ reviewer, threshold }: { reviewer: string; threshold: number }) {
  const load = useCallback(() => api.reidQueue(20), []);
  const { current, loading, error, next } = useQueue<ReidSummary>(load);
  const [item, setItem] = useState<ReidItem | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  useEffect(() => {
    setItem(null);
    if (current) api.reidItem(current.id).then(setItem, (e: Error) => setSaveError(e.message));
  }, [current]);

  const decide = useCallback(
    async (decision: "match" | "new" | "unsure", leopard?: string) => {
      if (!item || saving) return;
      setSaving(true);
      try {
        await api.reviewReid(item.id, decision, reviewer, leopard);
        setSaveError(null);
        next(item.id);
      } catch (e) {
        setSaveError((e as Error).message);
      } finally {
        setSaving(false);
      }
    },
    [item, next, reviewer, saving],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!item) return;
      const k = Number(e.key);
      if (k >= 1 && k <= item.candidates.length) void decide("match", item.candidates[k - 1].leopard);
      else if (e.key.toLowerCase() === "n") void decide("new");
      else if (e.key.toLowerCase() === "u") void decide("unsure");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [item, decide]);

  if (error) return <p className="error">Couldn't load the queue: {error}</p>;
  if (!current) return <p className="empty">{loading ? "Loading…" : "Nothing left to review."}</p>;
  if (!item) return <p className="empty">Loading…</p>;

  const suggestion = item.best_sim === null
    ? `No catalogued leopard has a photo of its ${item.viewpoint} flank yet.`
    : item.suggest_new
      ? `Model suggests a new leopard: best match ${item.best_sim.toFixed(2)} is below ${threshold.toFixed(2)}.`
      : `Model suggests ${shortId(item.candidates[0].leopard)} (similarity ${item.best_sim.toFixed(2)}).`;

  return (
    <section className="reid">
      <div className="reid-query">
        <figure className="photo">
          <img src={item.image_url} alt={`Leopard photo ${item.crop_id}`} />
          <figcaption>
            {item.viewpoint} flank · item {item.id}
          </figcaption>
        </figure>
        <div className="panel">
          <h2>Have we seen this leopard before?</h2>
          <p className={item.suggest_new ? "suggest new" : "suggest"}>{suggestion}</p>
          <p className="hint">Compare the spots on the same flank. Keys: 1–5 pick a leopard, N new, U unsure.</p>
          <div className="actions">
            <button className={item.suggest_new ? "primary" : ""} disabled={saving} onClick={() => void decide("new")}>
              <kbd>N</kbd> New leopard
            </button>
            <button disabled={saving} onClick={() => void decide("unsure")}>
              <kbd>U</kbd> Unsure
            </button>
          </div>
          {saveError && <p className="error">Not saved: {saveError}</p>}
        </div>
      </div>

      <ol className="candidates">
        {item.candidates.map((c) => (
          <li key={c.leopard}>
            <button className="candidate" disabled={saving} onClick={() => void decide("match", c.leopard)}>
              <img src={c.image_url} alt={`Catalogued leopard ${shortId(c.leopard)}`} />
              <span className="candidate-meta">
                <kbd>{c.rank}</kbd>
                <span className="mono">{shortId(c.leopard)}</span>
                <span className="sim">{c.sim.toFixed(2)}</span>
              </span>
            </button>
          </li>
        ))}
      </ol>
    </section>
  );
}
