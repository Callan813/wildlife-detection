// Database access. The model tables (species_items, reid_*, meta) are written by
// scripts/22_export_review.py; this server only reads them. The review tables below
// belong to the server and survive a re-export.

import { DatabaseSync } from "node:sqlite";

const REVIEW_SCHEMA = `
CREATE TABLE IF NOT EXISTS species_reviews (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  item_id INTEGER NOT NULL,
  label TEXT NOT NULL,
  reviewer TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX IF NOT EXISTS species_reviews_item ON species_reviews(item_id);
CREATE TABLE IF NOT EXISTS reid_reviews (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  item_id INTEGER NOT NULL,
  decision TEXT NOT NULL CHECK (decision IN ('match', 'new', 'unsure')),
  leopard TEXT,
  reviewer TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX IF NOT EXISTS reid_reviews_item ON reid_reviews(item_id);
`;

export function openDb(path: string): DatabaseSync {
  const db = new DatabaseSync(path);
  db.exec("PRAGMA journal_mode = WAL;");
  db.exec(REVIEW_SCHEMA);
  return db;
}

export function meta(db: DatabaseSync): Record<string, string> {
  const rows = db.prepare("SELECT key, value FROM meta").all() as { key: string; value: string }[];
  return Object.fromEntries(rows.map((r) => [r.key, r.value]));
}
