// HTTP API for the reviewer dashboard.
//
//   GET  /api/meta                    model names, threshold, export time
//   GET  /api/species/queue           unreviewed crops, least certain first, audit items mixed in
//   GET  /api/species/classes         the species groups a reviewer can pick
//   POST /api/species/:id/review      { label, reviewer? }
//   GET  /api/reid/queue              unreviewed leopard photos, least certain first, audit items mixed in
//   GET  /api/reid/items/:id          one photo with its top-k shortlist
//   GET  /api/reid/leopards           catalogue, for picking a leopard not on the shortlist
//   POST /api/reid/:id/review         { decision: match|new|unsure, leopard?, reviewer? }
//   GET  /api/stats                   progress, model vs reviewers, recalibrated threshold
//   GET  /images/:kind/:cropId        crop image (kind = species | reid)
//   GET  /images/frame/:id            full camera frame for species item :id

import path from "node:path";
import type { DatabaseSync } from "node:sqlite";
import express, { type NextFunction, type Request, type Response } from "express";
import { z } from "zod";
import { meta } from "./db.js";
import { interleave } from "./queue.js";
import { calibrate } from "./threshold.js";

export interface AppOptions {
  db: DatabaseSync;
  repoRoot: string; // image folders in meta are relative to this
}

const limitQuery = z.object({ limit: z.coerce.number().int().min(1).max(100).default(20) });
const idParam = z.object({ id: z.coerce.number().int().positive() });
const reviewer = z.string().trim().min(1).max(64).default("anonymous");
const speciesReview = z.object({ label: z.string().min(1), reviewer });
const reidReview = z
  .object({ decision: z.enum(["match", "new", "unsure"]), leopard: z.string().min(1).optional(), reviewer })
  .refine((r) => (r.decision === "match") === (r.leopard !== undefined), {
    message: "leopard is required for a match and not allowed otherwise",
  });

class HttpError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

export function createApp({ db, repoRoot }: AppOptions) {
  const app = express();
  app.use(express.json());
  const m = meta(db);
  const imageDirs: Record<string, string> = {
    species: path.resolve(repoRoot, m.species_images),
    reid: path.resolve(repoRoot, m.reid_images),
  };
  const classes: string[] = Object.keys(
    JSON.parse((db.prepare("SELECT probs FROM species_items LIMIT 1").get() as { probs: string }).probs),
  );
  const startThreshold = Number(m.new_threshold);

  app.get("/api/meta", (_req, res) => {
    res.json({ ...m, new_threshold: startThreshold });
  });

  // ---- species ----
  app.get("/api/species/classes", (_req, res) => res.json(classes));

  app.get("/api/species/queue", (req, res) => {
    const { limit } = limitQuery.parse(req.query);
    const select = `SELECT id, crop_id, location, detector_conf, pred, pred_prob, margin, probs, bbox, captured_at
      FROM species_items WHERE id NOT IN (SELECT item_id FROM species_reviews)`;
    const uncertain = db
      .prepare(`${select} AND audit_rank IS NULL ORDER BY margin, id LIMIT ?`)
      .all(limit) as unknown as SpeciesRow[];
    const audit = db
      .prepare(`${select} AND audit_rank IS NOT NULL ORDER BY audit_rank LIMIT ?`)
      .all(limit) as unknown as SpeciesRow[];
    res.json(interleave(uncertain, audit, limit).map(speciesItem));
  });

  app.post("/api/species/:id/review", (req, res) => {
    const { id } = idParam.parse(req.params);
    const body = speciesReview.parse(req.body);
    if (!classes.includes(body.label)) throw new HttpError(400, `unknown label ${body.label}`);
    if (!db.prepare("SELECT 1 FROM species_items WHERE id = ?").get(id)) throw new HttpError(404, "no such item");
    db.prepare("INSERT INTO species_reviews (item_id, label, reviewer) VALUES (?, ?, ?)").run(
      id,
      body.label,
      body.reviewer,
    );
    res.status(201).json({ ok: true });
  });

  // ---- re-ID ----
  app.get("/api/reid/queue", (req, res) => {
    const { limit } = limitQuery.parse(req.query);
    // Least certain first. Below the threshold the model says "new", and how sure it is
    // depends only on how far below. Above it, it's unsure if the best match is close to
    // the threshold or the top two leopards are close. Photos with no same-flank
    // catalogue match (best_sim NULL) go last: there is nothing to compare them with.
    const select = `SELECT id, crop_id, viewpoint, best_sim, margin, suggest_new FROM reid_items
      WHERE id NOT IN (SELECT item_id FROM reid_reviews)`;
    const uncertain = db
      .prepare(
        `${select} AND audit_rank IS NULL
         ORDER BY best_sim IS NULL,
                  CASE WHEN best_sim < :t THEN :t - best_sim ELSE MIN(IFNULL(margin, 1), best_sim - :t) END,
                  id
         LIMIT :limit`,
      )
      .all({ t: startThreshold, limit }) as unknown as ReidRow[];
    const audit = db
      .prepare(`${select} AND audit_rank IS NOT NULL ORDER BY audit_rank LIMIT ?`)
      .all(limit) as unknown as ReidRow[];
    res.json(interleave(uncertain, audit, limit).map((r) => reidSummary(r)));
  });

  app.get("/api/reid/items/:id", (req, res) => {
    const { id } = idParam.parse(req.params);
    const row = db
      .prepare("SELECT id, crop_id, viewpoint, best_sim, margin, suggest_new FROM reid_items WHERE id = ?")
      .get(id) as ReidRow | undefined;
    if (!row) throw new HttpError(404, "no such item");
    const candidates = db
      .prepare("SELECT rank, leopard, sim, example_crop_id FROM reid_candidates WHERE item_id = ? ORDER BY rank")
      .all(id) as { rank: number; leopard: string; sim: number; example_crop_id: string }[];
    res.json({
      ...reidSummary(row),
      candidates: candidates.map((c) => ({ ...c, image_url: `/images/reid/${encodeURIComponent(c.example_crop_id)}` })),
    });
  });

  app.get("/api/reid/leopards", (_req, res) => {
    const rows = db
      .prepare(
        `SELECT leopard, COUNT(*) AS photos, MIN(crop_id) AS example_crop_id FROM reid_catalogue
         GROUP BY leopard ORDER BY leopard`,
      )
      .all();
    res.json(rows);
  });

  app.post("/api/reid/:id/review", (req, res) => {
    const { id } = idParam.parse(req.params);
    const body = reidReview.parse(req.body);
    if (!db.prepare("SELECT 1 FROM reid_items WHERE id = ?").get(id)) throw new HttpError(404, "no such item");
    if (body.leopard && !db.prepare("SELECT 1 FROM reid_catalogue WHERE leopard = ?").get(body.leopard)) {
      throw new HttpError(400, `leopard ${body.leopard} is not in the catalogue`);
    }
    db.prepare("INSERT INTO reid_reviews (item_id, decision, leopard, reviewer) VALUES (?, ?, ?, ?)").run(
      id,
      body.decision,
      body.leopard ?? null,
      body.reviewer,
    );
    res.status(201).json({ ok: true });
  });

  // ---- stats ----
  app.get("/api/stats", (_req, res) => {
    res.json(stats(db, startThreshold));
  });

  // ---- images ----
  // Registered before the crop route so "frame" isn't read as a crop kind.
  app.get("/images/frame/:id", (req, res) => {
    const { id } = idParam.parse(req.params);
    const row = db.prepare("SELECT frame_path FROM species_items WHERE id = ?").get(id) as
      | { frame_path: string | null }
      | undefined;
    if (!row?.frame_path) throw new HttpError(404, "no frame for this item");
    // The path comes from the export, but still refuse anything outside the repo's data folder.
    const file = path.resolve(repoRoot, row.frame_path);
    if (!file.startsWith(path.resolve(repoRoot, "data") + path.sep)) throw new HttpError(404, "no frame for this item");
    res.sendFile(file, { maxAge: "1d" });
  });

  app.get("/images/:kind/:cropId", (req, res) => {
    const { kind, cropId } = req.params;
    if (kind !== "species" && kind !== "reid") throw new HttpError(404, "unknown image kind");
    // Only ids the database knows are served, so the URL can't reach other files.
    const known =
      kind === "species"
        ? db.prepare("SELECT 1 FROM species_items WHERE crop_id = ?").get(cropId)
        : db
            .prepare("SELECT 1 FROM reid_items WHERE crop_id = ? UNION SELECT 1 FROM reid_catalogue WHERE crop_id = ?")
            .get(cropId, cropId);
    if (!known) throw new HttpError(404, "no such image");
    res.sendFile(`${cropId}.jpg`, { root: imageDirs[kind], maxAge: "1d" });
  });

  app.use((err: unknown, _req: Request, res: Response, _next: NextFunction) => {
    if (err instanceof z.ZodError) {
      res.status(400).json({ error: "invalid request", issues: err.issues });
    } else if (err instanceof HttpError) {
      res.status(err.status).json({ error: err.message });
    } else {
      console.error(err);
      res.status(500).json({ error: "internal error" });
    }
  });
  return app;
}

interface SpeciesRow {
  id: number;
  crop_id: string;
  location: string;
  detector_conf: number;
  pred: string;
  pred_prob: number;
  margin: number;
  probs: string;
  bbox: string | null;
  captured_at: string | null;
}

function speciesItem(r: SpeciesRow) {
  const probs = Object.entries(JSON.parse(r.probs) as Record<string, number>)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([label, p]) => ({ label, p }));
  return {
    id: r.id,
    crop_id: r.crop_id,
    image_url: `/images/species/${encodeURIComponent(r.crop_id)}`,
    frame_url: `/images/frame/${r.id}`,
    bbox: r.bbox ? (JSON.parse(r.bbox) as [number, number, number, number]) : null, // normalised x1, y1, x2, y2
    captured_at: r.captured_at,
    location: r.location,
    detector_conf: r.detector_conf,
    pred: r.pred,
    pred_prob: r.pred_prob,
    margin: r.margin,
    top: probs,
  };
}

interface ReidRow {
  id: number;
  crop_id: string;
  viewpoint: string;
  best_sim: number | null;
  margin: number | null;
  suggest_new: number;
}

function reidSummary(r: ReidRow) {
  return {
    id: r.id,
    crop_id: r.crop_id,
    image_url: `/images/reid/${encodeURIComponent(r.crop_id)}`,
    viewpoint: r.viewpoint,
    best_sim: r.best_sim,
    margin: r.margin,
    suggest_new: r.suggest_new === 1,
  };
}

function stats(db: DatabaseSync, startThreshold: number) {
  const count = (sql: string) => (db.prepare(sql).get() as { n: number }).n;
  const share = (n: number, d: number) => (d ? n / d : null);
  // Latest decision per item counts, so a reviewer can correct a mistake by reviewing again.
  const sp = db
    .prepare(
      `SELECT s.pred, s.true_label, s.audit_rank IS NOT NULL AS audit, r.label
       FROM species_reviews r JOIN species_items s ON s.id = r.item_id
       WHERE r.id IN (SELECT MAX(id) FROM species_reviews GROUP BY item_id)`,
    )
    .all() as { pred: string; true_label: string; audit: number; label: string }[];
  const rr = db
    .prepare(
      `SELECT i.best_sim, i.suggest_new, i.truth_is_new, i.true_leopard, i.audit_rank IS NOT NULL AS audit,
              r.decision, r.leopard,
              (SELECT leopard FROM reid_candidates c WHERE c.item_id = i.id AND c.rank = 1) AS top1
       FROM reid_reviews r JOIN reid_items i ON i.id = r.item_id
       WHERE r.id IN (SELECT MAX(id) FROM reid_reviews GROUP BY item_id)`,
    )
    .all() as {
    best_sim: number | null;
    suggest_new: number;
    truth_is_new: number;
    true_leopard: string;
    audit: number;
    decision: string;
    leopard: string | null;
    top1: string | null;
  }[];

  // Model quality is measured on the random audit sample only; the uncertain-first items
  // are the model's hardest cases by construction.
  const spAudit = sp.filter((r) => r.audit);
  const spHard = sp.filter((r) => !r.audit);
  const decided = rr.filter((r) => r.decision !== "unsure");
  const rrAudit = decided.filter((r) => r.audit);
  const auditMatches = rrAudit.filter((r) => r.decision === "match");
  const calibration = calibrate(
    rrAudit.filter((r) => r.best_sim !== null).map((r) => ({ bestSim: r.best_sim!, isNew: r.decision === "new" })),
  );

  return {
    species: {
      total: count("SELECT COUNT(*) AS n FROM species_items"),
      reviewed: sp.length,
      audit_reviewed: spAudit.length,
      model_accuracy: share(spAudit.filter((r) => r.pred === r.label).length, spAudit.length),
      model_agrees_on_hardest: share(spHard.filter((r) => r.pred === r.label).length, spHard.length),
      // Simulation only: the export keeps the dataset's labels to check reviewers against.
      reviewer_agrees_with_dataset: share(sp.filter((r) => r.label === r.true_label).length, sp.length),
    },
    reid: {
      total: count("SELECT COUNT(*) AS n FROM reid_items"),
      reviewed: rr.length,
      audit_reviewed: rrAudit.length,
      unsure: rr.length - decided.length,
      top1_confirmed: share(auditMatches.filter((r) => r.leopard === r.top1).length, auditMatches.length),
      new_suggestion_agrees: share(
        rrAudit.filter((r) => (r.decision === "new") === (r.suggest_new === 1)).length,
        rrAudit.length,
      ),
      reviewer_agrees_with_dataset: share(
        decided.filter((r) => (r.decision === "new" ? r.truth_is_new === 1 : r.leopard === r.true_leopard)).length,
        decided.length,
      ),
      threshold: { start: startThreshold, recalibrated: calibration },
    },
  };
}
