import { useCallback, useEffect, useState } from "react";
import { api, type BurstFrame, label, pct, type SpeciesItem } from "./api";
import { useQueue } from "./useQueue";

// One crop at a time, least certain first. Keys 1-3 accept one of the model's top three,
// U marks it "not sure", or pick any group from the list. Left/right arrows step through
// the other frames of the same trigger burst: the animal often turns or moves into the open.
export function SpeciesReview({ reviewer }: { reviewer: string }) {
  const load = useCallback(() => api.speciesQueue(20), []);
  const { current, loading, error, next } = useQueue<SpeciesItem>(load);
  const [classes, setClasses] = useState<string[]>([]);
  const [other, setOther] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  // Which burst frame is on screen, tied to the item it was picked for, so a new item
  // starts on its boxed frame in the same render instead of one render later.
  const [view, setView] = useState<{ id: number; at: number } | null>(null);
  // The frame whose image has finished loading. Box and close-up wait for it: frames are
  // ~2.4 MB, and until one loads the browser still shows the previous photo.
  const [loadedUrl, setLoadedUrl] = useState<string | null>(null);

  const frames: BurstFrame[] = current?.burst.length
    ? current.burst
    : current
      ? [{ frame_num: 0, captured_at: current.captured_at, url: current.frame_url, current: true }]
      : [];
  const boxedAt = Math.max(
    0,
    frames.findIndex((f) => f.current),
  );
  const shown = view && current && view.id === current.id ? view.at : boxedAt;
  const viewing = frames[shown] ?? frames[boxedAt];
  const step = useCallback(
    (at: number) => current && setView({ id: current.id, at: Math.max(0, Math.min(frames.length - 1, at)) }),
    [current, frames.length],
  );

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
      else if (e.key.toLowerCase() === "u") void submit("unsure");
      else if (e.key === "ArrowLeft") step(shown - 1);
      else if (e.key === "ArrowRight") step(shown + 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [current, submit, step, shown]);

  if (error) return <p className="error">Couldn't load the queue: {error}</p>;
  if (!current || !viewing) return <p className="empty">{loading ? "Loading…" : "Nothing left to review."}</p>;
  const offset = shown - boxedAt;
  const ready = loadedUrl === viewing.url;
  const b = current.bbox;
  // Put the close-up in the corner away from the box so it never covers the animal.
  const insetSide = b && (b[0] + b[2]) / 2 > 0.5 ? "inset-left" : "";
  const insetEdge = b && (b[1] + b[3]) / 2 > 0.5 ? "inset-top" : "";

  return (
    <section className="review">
      <figure className="photo">
        {/* The whole frame with the detector's box: the crop alone is often too small to judge. */}
        <div className="frame">
          <img
            className={ready ? "" : "loading"}
            src={viewing.url}
            alt={`Camera frame for item ${current.id}`}
            onLoad={(e) => setLoadedUrl(e.currentTarget.getAttribute("src"))}
          />
          {ready && viewing.current && b && (
            <span
              className="box"
              style={{
                left: `${b[0] * 100}%`,
                top: `${b[1] * 100}%`,
                width: `${(b[2] - b[0]) * 100}%`,
                height: `${(b[3] - b[1]) * 100}%`,
              }}
            />
          )}
          {ready && viewing.current && (
            <img
              className={`inset ${insetSide} ${insetEdge}`}
              src={current.image_url}
              alt={`Close-up of the box, item ${current.id}`}
            />
          )}
          {!ready && <span className="frame-tag">Loading photo…</span>}
          {ready && !viewing.current && (
            <span className="frame-tag">
              {Math.abs(offset)} frame{Math.abs(offset) > 1 ? "s" : ""} {offset < 0 ? "before" : "after"}
            </span>
          )}
        </div>
        {frames.length > 1 && (
          <ol className="burst" aria-label="Other frames from this trigger">
            {frames.map((f, i) => (
              <li key={f.url}>
                <button
                  className={i === shown ? "active" : ""}
                  onClick={() => step(i)}
                  aria-label={f.current ? "Frame with the box" : `Frame ${f.frame_num}`}
                >
                  <img src={f.url} alt="" loading="lazy" />
                  {f.current && <span className="burst-mark">box</span>}
                </button>
              </li>
            ))}
          </ol>
        )}
        <figcaption>
          Camera {current.location}
          {viewing.captured_at && ` · ${viewing.captured_at.slice(0, 19)}`} · detector confidence{" "}
          {pct(current.detector_conf)} · item {current.id}
          {current.burst_size > 1 &&
            ` · frame ${frames[boxedAt].frame_num + 1} of ${current.burst_size} in this trigger (← → to step)`}
        </figcaption>
      </figure>

      <div className="panel">
        <h2>What animal is this?</h2>
        <p className="hint">
          The model is unsure: its top two guesses are {pct(current.margin, 1)} apart. Press 1–3 to pick one, or U
          if you can't tell.
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
        <div className="actions">
          <button disabled={saving} onClick={() => void submit("unsure")}>
            <kbd>U</kbd> Not sure
          </button>
        </div>
        {saveError && <p className="error">Not saved: {saveError}</p>}
      </div>
    </section>
  );
}
