import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { DatabaseSync } from "node:sqlite";
import request from "supertest";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createApp } from "../src/app.js";
import { openDb } from "../src/db.js";

// Same tables as scripts/22_export_review.py writes, with a handful of rows.
const EXPORT_SCHEMA = `
CREATE TABLE species_items (id INTEGER PRIMARY KEY, crop_id TEXT UNIQUE NOT NULL, seq_key TEXT, location TEXT,
  detector_conf REAL, pred TEXT NOT NULL, pred_prob REAL NOT NULL, margin REAL NOT NULL, probs TEXT NOT NULL,
  true_label TEXT, frame_path TEXT, bbox TEXT, captured_at TEXT, audit_rank INTEGER);
CREATE TABLE reid_catalogue (crop_id TEXT PRIMARY KEY, leopard TEXT NOT NULL, viewpoint TEXT NOT NULL,
  encounter TEXT NOT NULL);
CREATE TABLE reid_items (id INTEGER PRIMARY KEY, crop_id TEXT UNIQUE NOT NULL, viewpoint TEXT NOT NULL,
  encounter TEXT NOT NULL, best_sim REAL, margin REAL, suggest_new INTEGER NOT NULL, true_leopard TEXT,
  truth_is_new INTEGER, audit_rank INTEGER);
CREATE TABLE reid_candidates (item_id INTEGER NOT NULL, rank INTEGER NOT NULL, leopard TEXT NOT NULL,
  sim REAL NOT NULL, example_crop_id TEXT NOT NULL, PRIMARY KEY (item_id, rank));
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
INSERT INTO meta VALUES ('new_threshold', '0.5'), ('species_images', 'img/species'), ('reid_images', 'img/reid');
INSERT INTO species_items VALUES
  (1, 'sure', 's1', 'loc1', 0.9, 'civet', 0.9, 0.85, '{"civet":0.9,"sambar":0.05,"macaque":0.05}', 'civet',
   'data/frames/f1.jpg', '[0.1, 0.2, 0.3, 0.4]', '2017-06-30 03:07:20', 1),
  (2, 'unsure', 's2', 'loc1', 0.5, 'sambar', 0.45, 0.05, '{"civet":0.1,"sambar":0.45,"macaque":0.4}', 'macaque',
   '../secret.txt', NULL, NULL, NULL);
INSERT INTO reid_catalogue VALUES ('c1', 'L1', 'left', 'e1'), ('c2', 'L2', 'left', 'e2');
INSERT INTO reid_items VALUES
  (1, 'q1', 'left', 'e3', 0.8, 0.3, 0, 'L1', 0, 1),
  (2, 'q2', 'left', 'e4', 0.52, 0.02, 0, 'L9', 1, NULL),
  (3, 'q3', 'right', 'e5', NULL, NULL, 1, 'L1', 1, NULL),
  (4, 'q4', 'left', 'e6', 0.1, 0.001, 1, 'L5', 1, NULL);
INSERT INTO reid_candidates VALUES (1, 1, 'L1', 0.8, 'c1'), (1, 2, 'L2', 0.5, 'c2'),
  (2, 1, 'L2', 0.52, 'c2'), (2, 2, 'L1', 0.5, 'c1');
`;

let app: ReturnType<typeof createApp>;
let db: DatabaseSync;
let root: string;

beforeEach(() => {
  root = fs.mkdtempSync(path.join(os.tmpdir(), "review-"));
  const file = path.join(root, "review.db");
  const seed = new DatabaseSync(file);
  seed.exec(EXPORT_SCHEMA);
  seed.close();
  fs.mkdirSync(path.join(root, "img/species"), { recursive: true });
  fs.writeFileSync(path.join(root, "img/species/sure.jpg"), "jpeg bytes");
  fs.writeFileSync(path.join(root, "secret.txt"), "not an image");
  fs.mkdirSync(path.join(root, "data/frames"), { recursive: true });
  fs.writeFileSync(path.join(root, "data/frames/f1.jpg"), "frame bytes");
  db = openDb(file);
  app = createApp({ db, repoRoot: root });
});

afterEach(() => {
  db.close();
  fs.rmSync(root, { recursive: true, force: true });
});

describe("species", () => {
  it("queues the least certain crop first and hides reviewed ones", async () => {
    const q = await request(app).get("/api/species/queue").expect(200);
    expect(q.body.map((i: { crop_id: string }) => i.crop_id)).toEqual(["unsure", "sure"]);
    expect(q.body[0].top[0]).toEqual({ label: "sambar", p: 0.45 });
    expect(q.body[1]).toMatchObject({ frame_url: "/images/frame/1", bbox: [0.1, 0.2, 0.3, 0.4] });

    await request(app).post("/api/species/2/review").send({ label: "macaque" }).expect(201);
    const after = await request(app).get("/api/species/queue").expect(200);
    expect(after.body.map((i: { crop_id: string }) => i.crop_id)).toEqual(["sure"]);
  });

  it("rejects unknown labels and items", async () => {
    await request(app).post("/api/species/2/review").send({ label: "tiger" }).expect(400);
    await request(app).post("/api/species/99/review").send({ label: "civet" }).expect(404);
    await request(app).post("/api/species/abc/review").send({ label: "civet" }).expect(400);
  });
});

describe("re-id", () => {
  it("puts the item near the threshold first and leaves no-match items last", async () => {
    const q = await request(app).get("/api/reid/queue").expect(200);
    // Item 4 has two near-tied candidates, but its best match is far below the threshold:
    // the model is sure it's new, so it isn't urgent. Item 1 is an audit item; with only
    // three uncertain items it fills in after them.
    expect(q.body.map((i: { id: number }) => i.id)).toEqual([2, 4, 3, 1]);
  });

  it("returns the shortlist with images", async () => {
    const r = await request(app).get("/api/reid/items/1").expect(200);
    expect(r.body.candidates).toHaveLength(2);
    expect(r.body.candidates[0]).toMatchObject({ rank: 1, leopard: "L1", image_url: "/images/reid/c1" });
  });

  it("validates decisions", async () => {
    await request(app).post("/api/reid/1/review").send({ decision: "match" }).expect(400); // no leopard
    await request(app).post("/api/reid/1/review").send({ decision: "new", leopard: "L1" }).expect(400);
    await request(app).post("/api/reid/1/review").send({ decision: "match", leopard: "L7" }).expect(400);
    await request(app).post("/api/reid/1/review").send({ decision: "match", leopard: "L1" }).expect(201);
    await request(app).post("/api/reid/3/review").send({ decision: "new" }).expect(201);
  });
});

describe("stats", () => {
  it("measures the model on audit items only, using each item's latest decision", async () => {
    await request(app).post("/api/species/2/review").send({ label: "civet" }).expect(201);
    await request(app).post("/api/species/2/review").send({ label: "macaque" }).expect(201); // correction
    await request(app).post("/api/reid/1/review").send({ decision: "match", leopard: "L1" }).expect(201); // audit
    await request(app).post("/api/reid/2/review").send({ decision: "new" }).expect(201); // not audit

    const s = (await request(app).get("/api/stats").expect(200)).body;
    expect(s.species).toMatchObject({
      reviewed: 1,
      audit_reviewed: 0,
      model_accuracy: null, // item 2 isn't an audit item
      model_agrees_on_hardest: 0,
      reviewer_agrees_with_dataset: 1,
    });
    // Item 2's "new" disagrees with the model, but it isn't an audit item, so it doesn't count.
    expect(s.reid).toMatchObject({ reviewed: 2, audit_reviewed: 1, top1_confirmed: 1, new_suggestion_agrees: 1 });
    expect(s.reid.threshold).toEqual({ start: 0.5, recalibrated: null }); // too few reviews yet
  });
});

describe("images", () => {
  it("serves known crops only", async () => {
    await request(app).get("/images/species/sure").expect(200);
    await request(app).get("/images/species/nope").expect(404);
    await request(app).get("/images/species/..%2Fsecret").expect(404);
    await request(app).get("/images/other/sure").expect(404);
  });

  it("serves full frames by item id, only from the data folder", async () => {
    await request(app).get("/images/frame/1").expect(200);
    await request(app).get("/images/frame/2").expect(404); // path points outside data/
    await request(app).get("/images/frame/99").expect(404);
  });
});
