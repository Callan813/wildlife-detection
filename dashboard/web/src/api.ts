// Typed calls to the review API (dashboard/server). Shapes mirror server/src/app.ts.

export interface SpeciesItem {
  id: number;
  crop_id: string;
  image_url: string;
  frame_url: string;
  bbox: [number, number, number, number] | null; // normalised x1, y1, x2, y2
  captured_at: string | null;
  location: string;
  detector_conf: number;
  pred: string;
  pred_prob: number;
  margin: number;
  top: { label: string; p: number }[];
  burst: BurstFrame[]; // up to 3 frames either side of this one, in order; empty if unknown
  burst_size: number; // frames in the whole trigger burst
}

export interface BurstFrame {
  frame_num: number;
  captured_at: string | null;
  url: string;
  current: boolean; // the frame the box was drawn on
}

export interface ReidSummary {
  id: number;
  crop_id: string;
  image_url: string;
  viewpoint: string;
  best_sim: number | null;
  margin: number | null;
  suggest_new: boolean;
}

export interface Candidate {
  rank: number;
  leopard: string;
  sim: number;
  example_crop_id: string;
  image_url: string;
}

export interface ReidItem extends ReidSummary {
  candidates: Candidate[];
}

export interface Meta {
  exported_at: string;
  species_model: string;
  reid_model: string;
  new_threshold: number;
}

export interface Stats {
  species: {
    total: number;
    reviewed: number;
    audit_reviewed: number;
    unsure: number;
    model_accuracy: number | null;
    model_agrees_on_hardest: number | null;
    reviewer_agrees_with_dataset: number | null;
  };
  reid: {
    total: number;
    reviewed: number;
    audit_reviewed: number;
    unsure: number;
    top1_confirmed: number | null;
    new_suggestion_agrees: number | null;
    reviewer_agrees_with_dataset: number | null;
    threshold: {
      start: number;
      recalibrated: { threshold: number; balancedAccuracy: number; matches: number; news: number } | null;
    };
  };
}

async function call<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error ?? `${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

const post = (url: string, body: unknown) => call<{ ok: true }>(url, { method: "POST", body: JSON.stringify(body) });

export const api = {
  meta: () => call<Meta>("/api/meta"),
  stats: () => call<Stats>("/api/stats"),
  speciesClasses: () => call<string[]>("/api/species/classes"),
  speciesQueue: (limit = 20) => call<SpeciesItem[]>(`/api/species/queue?limit=${limit}`),
  reviewSpecies: (id: number, label: string, reviewer: string) =>
    post(`/api/species/${id}/review`, { label, reviewer }),
  reidQueue: (limit = 20) => call<ReidSummary[]>(`/api/reid/queue?limit=${limit}`),
  reidItem: (id: number) => call<ReidItem>(`/api/reid/items/${id}`),
  reviewReid: (id: number, decision: "match" | "new" | "unsure", reviewer: string, leopard?: string) =>
    post(`/api/reid/${id}/review`, { decision, leopard, reviewer }),
};

// Leopard ids are UUIDs; reviewers only need enough to tell them apart.
export const shortId = (leopard: string) => leopard.slice(0, 8);
export const pct = (v: number | null, digits = 0) => (v === null ? "–" : `${(v * 100).toFixed(digits)}%`);
export const label = (s: string) => {
  const words = s.replace(/_/g, " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
};
