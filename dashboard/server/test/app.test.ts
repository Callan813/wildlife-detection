import fs from "node:fs";
import http from "node:http";
import type { AddressInfo } from "node:net";
import os from "node:os";
import path from "node:path";
import { DatabaseSync } from "node:sqlite";
import request from "supertest";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createApp } from "../src/app.js";
import { openDb } from "../src/db.js";

// Same tables as scripts/22_export_review.py writes, with a handful of rows.
const EXPORT_SCHEMA = `
CREATE TABLE species_items (id INTEGER PRIMARY KEY, crop_id TEXT UNIQUE NOT NULL, image_id TEXT, seq_key TEXT,
  location TEXT, detector_conf REAL, pred TEXT NOT NULL, pred_prob REAL NOT NULL, margin REAL NOT NULL,
  probs TEXT NOT NULL, true_label TEXT, frame_path TEXT, bbox TEXT, captured_at TEXT, audit_rank INTEGER,
  phantom INTEGER NOT NULL DEFAULT 0);
CREATE TABLE species_frames (id INTEGER PRIMARY KEY, image_id TEXT UNIQUE NOT NULL, seq_key TEXT NOT NULL,
  frame_num INTEGER NOT NULL, captured_at TEXT, frame_path TEXT NOT NULL);
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
  (1, 'sure', 'f1', 's1', 'loc1', 0.9, 'civet', 0.9, 0.85, '{"civet":0.9,"sambar":0.05,"macaque":0.05}', 'civet',
   'data/frames/f1.jpg', '[0.1, 0.2, 0.3, 0.4]', '2017-06-30 03:07:20', 1, 0),
  (2, 'unsure', 'g1', 's2', 'loc1', 0.5, 'sambar', 0.45, 0.05, '{"civet":0.1,"sambar":0.45,"macaque":0.4}', 'macaque',
   '../secret.txt', NULL, NULL, NULL, 0),
  (3, 'strip', 'h1', 's3', 'loc1', 0.05, 'civet', 0.4, 0.01, '{"civet":0.4,"sambar":0.39,"macaque":0.21}', 'civet',
   'data/frames/f1.jpg', '[0.0, 0.0, 0.99, 0.0003]', NULL, NULL, 1);
-- Burst s1 has five frames (stored out of order); the item is the middle one. s2's frame
-- has no row here, as if the manifest lost it.
INSERT INTO species_frames VALUES
  (10, 'f3', 's1', 3, NULL, 'data/frames/f3.jpg'), (11, 'f0', 's1', 0, NULL, 'data/frames/f0.jpg'),
  (12, 'f1', 's1', 1, '2017-06-30 03:07:20', 'data/frames/f1.jpg'), (13, 'f2', 's1', 2, NULL, 'data/frames/f2.jpg'),
  (14, 'f4', 's1', 4, NULL, 'data/frames/f4.jpg'), (15, 'x', 's9', 0, NULL, '../secret.txt');
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
  for (const f of ["f0", "f1", "f2", "f3", "f4"]) fs.writeFileSync(path.join(root, `data/frames/${f}.jpg`), "frame bytes");
  db = openDb(file);
  app = createApp({ db, repoRoot: root });
});

afterEach(() => {
  db.close();
  fs.rmSync(root, { recursive: true, force: true });
});

describe("species", () => {
  it("leaves padding phantoms out of the queue and stats", async () => {
    // Item 3 is the least certain of all, but its box is a 1-px strip along the frame edge.
    const q = await request(app).get("/api/species/queue").expect(200);
    expect(q.body.map((i: { id: number }) => i.id)).not.toContain(3);
    await request(app).post("/api/species/3/review").send({ label: "civet" }).expect(201); // made before the flag
    const s = (await request(app).get("/api/stats").expect(200)).body.species;
    expect(s).toMatchObject({ total: 2, reviewed: 0 });
  });

  it("queues the least certain crop first and hides reviewed ones", async () => {
    const q = await request(app).get("/api/species/queue").expect(200);
    expect(q.body.map((i: { crop_id: string }) => i.crop_id)).toEqual(["unsure", "sure"]);
    expect(q.body[0].top[0]).toEqual({ label: "sambar", p: 0.45 });
    expect(q.body[1]).toMatchObject({ frame_url: "/images/frame/1", bbox: [0.1, 0.2, 0.3, 0.4] });

    await request(app).post("/api/species/2/review").send({ label: "macaque" }).expect(201);
    const after = await request(app).get("/api/species/queue").expect(200);
    expect(after.body.map((i: { crop_id: string }) => i.crop_id)).toEqual(["sure"]);
  });

  it("returns the frames around the boxed one, in burst order", async () => {
    const q = await request(app).get("/api/species/queue").expect(200);
    const sure = q.body.find((i: { id: number }) => i.id === 1);
    expect(sure.burst_size).toBe(5);
    expect(sure.burst.map((f: { frame_num: number }) => f.frame_num)).toEqual([0, 1, 2, 3, 4]);
    expect(sure.burst[1]).toMatchObject({ current: true, url: "/images/frame/1" });
    expect(sure.burst[0]).toMatchObject({ current: false, url: "/images/burst/11" });
    const lost = q.body.find((i: { id: number }) => i.id === 2);
    expect(lost).toMatchObject({ burst: [], burst_size: 0 });
  });

  it("takes 'unsure' as a review but leaves it out of the accuracy numbers", async () => {
    await request(app).post("/api/species/1/review").send({ label: "unsure" }).expect(201);
    const q = await request(app).get("/api/species/queue").expect(200);
    expect(q.body.map((i: { id: number }) => i.id)).toEqual([2]);
    const s = (await request(app).get("/api/stats").expect(200)).body.species;
    expect(s).toMatchObject({ reviewed: 1, unsure: 1, audit_reviewed: 0, model_accuracy: null });
    expect(s.reviewer_agrees_with_dataset).toBeNull();
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

  it("serves burst frames by frame id, only from the data folder", async () => {
    await request(app).get("/images/burst/11").expect(200);
    await request(app).get("/images/burst/15").expect(404); // path points outside data/
    await request(app).get("/images/burst/99").expect(404);
  });
});

describe("search", () => {
  // Stands in for scripts/31_search_server.py and records what it was asked.
  let upstream: http.Server;
  let seen: string[];
  let searchUrl: string;

  beforeEach(async () => {
    seen = [];
    upstream = http.createServer((req, res) => {
      seen.push(req.url ?? "");
      if (req.url?.startsWith("/images/")) {
        res.writeHead(200, { "Content-Type": "image/jpeg", "Cache-Control": "max-age=86400" }).end("jpeg");
      } else {
        res.writeHead(req.url?.includes("bad") ? 404 : 200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ results: [], url: req.url }));
      }
    });
    await new Promise<void>((resolve) => upstream.listen(0, "127.0.0.1", resolve));
    searchUrl = `http://127.0.0.1:${(upstream.address() as AddressInfo).port}`;
  });

  afterEach(() => new Promise<void>((resolve) => upstream.close(() => resolve())));

  it("forwards queries and passes the status on", async () => {
    const a = createApp({ db, repoRoot: root, searchUrl });
    const res = await request(a).get("/api/search?q=civet%20at%20night&k=5").expect(200);
    expect(res.body.url).toBe("/api/search?q=civet%20at%20night&k=5");
    await request(a).get("/api/similar?event=bad").expect(404);
    expect(seen).toEqual(["/api/search?q=civet%20at%20night&k=5", "/api/similar?event=bad"]);
  });

  it("forwards images with their type and cache header, and checks the path", async () => {
    const a = createApp({ db, repoRoot: root, searchUrl });
    const res = await request(a).get("/images/search/crop/abc-1_0").expect(200);
    expect(res.headers["content-type"]).toBe("image/jpeg");
    expect(res.headers["cache-control"]).toBe("max-age=86400");
    await request(a).get("/images/search/other/abc").expect(400);
    await request(a).get("/images/search/frame/..%2F..%2Fsecret").expect(400);
    expect(seen).toEqual(["/images/search/crop/abc-1_0"]);
  });

  it("answers 503 when the search server is down or not configured", async () => {
    const down = createApp({ db, repoRoot: root, searchUrl: "http://127.0.0.1:9" });
    const res = await request(down).get("/api/search?q=civet").expect(503);
    expect(res.body.error).toMatch(/31_search_server/);
    await request(app).get("/api/search?q=civet").expect(503);
  });
});
