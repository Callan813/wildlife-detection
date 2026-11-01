import { type FormEvent, useEffect, useState } from "react";
import { api, label, pct, type SearchResponse, type SearchResult } from "./api";

const EXAMPLES = [
  "civets at night",
  "two wild pigs",
  "macaque in daylight",
  "animal drinking water",
  "deer close to the camera",
  "empty photos on camera 162",
];

// How the query was read: the filters it became, and what was left for the image embeddings.
function Reading({ r }: { r: SearchResponse }) {
  const q = r.query;
  const chips: string[] = [
    ...q.species.map((s) => label(s)),
    ...(q.time ? [q.time] : []),
    ...(q.empty ? ["empty"] : []),
    ...(q.animal ? ["any animal"] : []),
    ...(q.min_count ? [`${q.min_count}+ animals`] : []),
    ...q.cameras,
    ...q.years.map(String),
    ...q.months.map((m) => new Date(2000, m - 1).toLocaleString(undefined, { month: "long" })),
  ];
  return (
    <p className="reading">
      {chips.map((c) => (
        <span key={c} className="chip">
          {c}
        </span>
      ))}
      {q.rest && <span className="chip meaning">“{q.rest}”</span>}
      <span className="hint">
        {r.matches} events match · ranked by {r.query.ranked_by}
      </span>
    </p>
  );
}

function Result({ r, onSimilar }: { r: SearchResult; onSimilar: () => void }) {
  const [loaded, setLoaded] = useState(false);
  const b = r.bbox;
  return (
    <li className="result">
      <a className="result-frame" href={r.frame_url} target="_blank" rel="noreferrer">
        <img src={r.frame_url} alt={`Event ${r.event}`} loading="lazy" onLoad={() => setLoaded(true)} />
        {loaded && b && (
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
        {r.night && <span className="frame-tag night">night</span>}
      </a>
      <div className="result-meta">
        <strong>{r.species ? label(r.species) : "No animal"}</strong>
        {r.species && <span className="hint">{pct(r.species_prob)}</span>}
        <span className="hint result-where">
          {r.location} · {new Date(r.start).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })}
          {r.frames > 1 && ` · ${r.frames} frames`}
        </span>
        <button className="similar" onClick={onSimilar}>
          More like this
        </button>
      </div>
    </li>
  );
}

// ?tab=search&q=… opens with that search, so a search can be shared as a link.
const initial = new URLSearchParams(window.location.search).get("q") ?? "";

export function SearchView() {
  const [text, setText] = useState(initial);
  const [res, setRes] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const run = async (p: Promise<SearchResponse>) => {
    setLoading(true);
    setError(null);
    try {
      setRes(await p);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };
  const search = (q: string) => {
    setText(q);
    if (q.trim()) void run(api.search(q.trim()));
  };
  useEffect(() => {
    if (initial.trim()) void run(api.search(initial.trim()));
  }, []);
  const submit = (e: FormEvent) => {
    e.preventDefault();
    search(text);
  };

  return (
    <section className="search">
      <form className="search-bar" onSubmit={submit}>
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Search the photos, e.g. civets at night"
          maxLength={200}
          aria-label="Search"
        />
        <button className="primary" disabled={loading || !text.trim()}>
          {loading ? "Searching…" : "Search"}
        </button>
      </form>
      {!res && !error && (
        <p className="examples">
          Try:{" "}
          {EXAMPLES.map((x) => (
            <button key={x} className="chip-button" onClick={() => search(x)}>
              {x}
            </button>
          ))}
        </p>
      )}
      {error && <p className="error">Search failed: {error}</p>}
      {res && (
        <>
          {res.query.ranked_by === "looks like" ? (
            <p className="reading">
              <span className="hint">Events that look like the one you picked</span>
            </p>
          ) : (
            <Reading r={res} />
          )}
          {res.results.length === 0 ? (
            <p className="empty">Nothing matches.</p>
          ) : (
            <ol className="results">
              {res.results.map((r) => (
                <Result key={r.event} r={r} onSimilar={() => void run(api.similar(r.event))} />
              ))}
            </ol>
          )}
        </>
      )}
    </section>
  );
}
