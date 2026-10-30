import { useEffect, useState } from "react";
import { api, type Meta } from "./api";
import { ReidReview } from "./ReidReview";
import { SpeciesReview } from "./SpeciesReview";
import { StatsView } from "./StatsView";

type Tab = "species" | "leopards" | "stats";
const TABS: { id: Tab; name: string }[] = [
  { id: "species", name: "Species" },
  { id: "leopards", name: "Leopards" },
  { id: "stats", name: "Stats" },
];

function stored(key: string, fallback: string): string {
  try {
    return localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
}

function store(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // private mode etc.: the name just won't be remembered
  }
}

// ?reviewer=…&tab=… preset the name and screen, e.g. a link sent to each reviewer.
const params = new URLSearchParams(window.location.search);
const isTab = (t: string | null): t is Tab => TABS.some((x) => x.id === t);

export function App() {
  const [tab, setTab] = useState<Tab>(() => {
    const t = params.get("tab") ?? stored("tab", "species");
    return isTab(t) ? t : "species";
  });
  const [reviewer, setReviewer] = useState(() => params.get("reviewer") ?? stored("reviewer", ""));
  const [meta, setMeta] = useState<Meta | null>(null);

  useEffect(() => {
    api.meta().then(setMeta, () => setMeta(null));
  }, []);
  useEffect(() => store("tab", tab), [tab]);
  useEffect(() => store("reviewer", reviewer), [reviewer]);

  return (
    <div className="app">
      <header>
        <h1>Ghats Review</h1>
        <nav>
          {TABS.map((t) => (
            <button key={t.id} className={t.id === tab ? "tab active" : "tab"} onClick={() => setTab(t.id)}>
              {t.name}
            </button>
          ))}
        </nav>
        <label className="reviewer">
          Reviewer
          <input value={reviewer} onChange={(e) => setReviewer(e.target.value)} placeholder="your name" />
        </label>
      </header>

      <main>
        {!reviewer.trim() && tab !== "stats" ? (
          <p className="empty">Enter your name at the top right to start reviewing.</p>
        ) : tab === "species" ? (
          <SpeciesReview reviewer={reviewer.trim()} />
        ) : tab === "leopards" ? (
          <ReidReview reviewer={reviewer.trim()} threshold={meta?.new_threshold ?? 0.5} />
        ) : (
          <StatsView />
        )}
      </main>

      {meta && (
        <footer>
          Predictions exported {new Date(meta.exported_at).toLocaleString()} · species: {meta.species_model}
        </footer>
      )}
    </div>
  );
}
