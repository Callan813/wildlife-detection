import { useCallback, useEffect, useState } from "react";
import { api, label, pct, type SpeciesItem } from "./api";
import { useQueue } from "./useQueue";

// One crop at a time, least certain first. Keys 1-3 accept one of the model's top three,
// or pick any group from the list.
export function SpeciesReview({ reviewer }: { reviewer: string }) {
  const load = useCallback(() => api.speciesQueue(20), []);
  const { current, loading, error, next } = useQueue<SpeciesItem>(load);
  const [classes, setClasses] = useState<string[]>([]);
  const [other, setOther] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  useEffect(() => {
    api.speciesClasses().then(setClasses, () => setClasses([]));
  }, []);

  const submit = useCallback(
    async (choice: string) => {
      if (!current || saving) return;
      setSaving(true);
      try {
        await api.reviewSpecies(current.id, choice, reviewer);
        setOther("");
        setSaveError(null);
        next(current.id);
      } catch (e) {
        setSaveError((e as Error).message);
      } finally {
        setSaving(false);
      }
    },
    [current, next, reviewer, saving],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!current || e.target instanceof HTMLSelectElement || e.target instanceof HTMLInputElement) return;
      const k = Number(e.key);
      if (k >= 1 && k <= current.top.length) void submit(current.top[k - 1].label);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [current, submit]);

  if (error) return <p className="error">Couldn't load the queue: {error}</p>;
  if (!current) return <p className="empty">{loading ? "Loading…" : "Nothing left to review."}</p>;

  return (
    <section className="review">
      <figure className="photo">
        {/* The whole frame with the detector's box: the crop alone is often too small to judge. */}
        <div className="frame">
          <img src={current.frame_url} alt={`Camera frame for item ${current.id}`} />
          {current.bbox && (
            <span
              className="box"
              style={{
                left: `${current.bbox[0] * 100}%`,
                top: `${current.bbox[1] * 100}%`,
                width: `${(current.bbox[2] - current.bbox[0]) * 100}%`,
                height: `${(current.bbox[3] - current.bbox[1]) * 100}%`,
              }}
            />
          )}
          <img className="inset" src={current.image_url} alt={`Close-up of the box, item ${current.id}`} />
        </div>
        <figcaption>
          Camera {current.location}
          {current.captured_at && ` · ${current.captured_at.slice(0, 16)}`} · detector confidence{" "}
          {pct(current.detector_conf)} · item {current.id}
        </figcaption>
      </figure>

      <div className="panel">
        <h2>What animal is this?</h2>
        <p className="hint">
          The model is unsure: its top two guesses are {pct(current.margin, 1)} apart. Press 1–3 to pick one.
        </p>
        <ol className="choices">
          {current.top.map((t, i) => (
            <li key={t.label}>
              <button className="choice" disabled={saving} onClick={() => void submit(t.label)}>
                <kbd>{i + 1}</kbd>
                <span className="choice-label">{label(t.label)}</span>
                <span className="bar" aria-hidden>
                  <span style={{ width: `${t.p * 100}%` }} />
                </span>
                <span className="prob">{pct(t.p)}</span>
              </button>
            </li>
          ))}
        </ol>
        <div className="other">
          <select value={other} onChange={(e) => setOther(e.target.value)} aria-label="Another group">
            <option value="">Something else…</option>
            {classes.map((c) => (
              <option key={c} value={c}>
                {label(c)}
              </option>
            ))}
          </select>
          <button disabled={!other || saving} onClick={() => void submit(other)}>
            Save
          </button>
        </div>
        {saveError && <p className="error">Not saved: {saveError}</p>}
      </div>
    </section>
  );
}
