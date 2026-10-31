"""Phase 5, step 1: write the review database the dashboard reads.

    python scripts/22_export_review.py --config configs/phase5.yaml

Models stay in Python; the dashboard (Node.js API + React UI in dashboard/) only reads
their outputs and records decisions. This script builds outputs/phase5/review.db:

  species_items    one row per test crop: predicted group, probabilities, margin, and the
                   full frame + box (a small crop alone is often too little to judge)
                   Crops whose box is a padding phantom (a strip a few pixels tall along the
                   frame edge, see the Phase 1 padding-box bug) are kept, so item ids stay
                   stable, but flagged: the dashboard leaves them out of queue and stats.
  species_frames   every frame of the trigger bursts those crops come from, so a reviewer
                   can look at the shots just before and after (the animal often turns
                   or steps into the open)
  reid_catalogue   crops of the leopards already known when review starts
  reid_items       crops to review: best similarity, margin, "new leopard" suggestion
  reid_candidates  top-k catalogued leopards per item, with the best-matching photo
  meta             model names, threshold, counts

Each item also gets an audit_rank: a random 20% (audit_frac) are an audit sample, in a
random order; NULL for the rest. The dashboard mixes audit items into the queue and
measures the models on them only, since the uncertain-first items are a biased sample.

Ground truth is stored too (true_label, true_leopard, truth_is_new) so the dashboard can
report how the model and the reviewers compare. A real deployment wouldn't have it.

The model tables are rebuilt on every run; the dashboard's own tables (reviews) are left
alone, so re-exporting doesn't wipe decisions already made.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ghats.config import load_config
from ghats.review import audit_ranks, shortlists, species_uncertainty, split_catalogue

SCHEMA = """
DROP TABLE IF EXISTS species_items;
DROP TABLE IF EXISTS species_frames;
DROP TABLE IF EXISTS reid_catalogue;
DROP TABLE IF EXISTS reid_items;
DROP TABLE IF EXISTS reid_candidates;
DROP TABLE IF EXISTS meta;
CREATE TABLE species_items (
  id INTEGER PRIMARY KEY, crop_id TEXT UNIQUE NOT NULL, image_id TEXT, seq_key TEXT, location TEXT,
  detector_conf REAL, pred TEXT NOT NULL, pred_prob REAL NOT NULL, margin REAL NOT NULL,
  probs TEXT NOT NULL, true_label TEXT, frame_path TEXT, bbox TEXT, captured_at TEXT, audit_rank INTEGER,
  phantom INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE species_frames (
  id INTEGER PRIMARY KEY, image_id TEXT UNIQUE NOT NULL, seq_key TEXT NOT NULL, frame_num INTEGER NOT NULL,
  captured_at TEXT, frame_path TEXT NOT NULL
);
CREATE INDEX species_frames_seq ON species_frames(seq_key, frame_num);
CREATE TABLE reid_catalogue (
  crop_id TEXT PRIMARY KEY, leopard TEXT NOT NULL, viewpoint TEXT NOT NULL, encounter TEXT NOT NULL
);
CREATE TABLE reid_items (
  id INTEGER PRIMARY KEY, crop_id TEXT UNIQUE NOT NULL, viewpoint TEXT NOT NULL, encounter TEXT NOT NULL,
  best_sim REAL, margin REAL, suggest_new INTEGER NOT NULL, true_leopard TEXT, truth_is_new INTEGER,
  audit_rank INTEGER
);
CREATE TABLE reid_candidates (
  item_id INTEGER NOT NULL REFERENCES reid_items(id), rank INTEGER NOT NULL, leopard TEXT NOT NULL,
  sim REAL NOT NULL, example_crop_id TEXT NOT NULL, PRIMARY KEY (item_id, rank)
);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def species_rows(cfg: dict) -> pd.DataFrame:
    p = pd.read_csv(cfg["predictions"])
    p = p[p["split"] == cfg["split"]].reset_index(drop=True)
    boxes = pd.read_csv(cfg["crops"], usecols=["crop_id", "bbox"]).set_index("crop_id")["bbox"]
    frames = pd.read_csv(cfg["manifest"], usecols=["image_id", "local_path", "datetime"]).set_index("image_id")
    frame = frames.reindex(p["image_id"])
    groups = [c[2:] for c in p.columns if c.startswith("p_")]
    probs = p[[f"p_{g}" for g in groups]].to_numpy()
    top, margin = species_uncertainty(probs)
    return pd.DataFrame(
        {
            "crop_id": p["crop_id"],
            "image_id": p["image_id"],
            "seq_key": p["seq_key"],
            "location": p["location"],
            "detector_conf": p["conf"],
            "pred": [groups[i] for i in top],
            "pred_prob": probs.max(1),
            "margin": margin,
            "probs": [json.dumps({g: round(float(v), 4) for g, v in zip(groups, row)}) for row in probs],
            "true_label": p["label"],
            "frame_path": [Path(f).as_posix() if isinstance(f, str) else None for f in frame["local_path"]],
            "bbox": p["crop_id"].map(boxes).to_numpy(),
            "captured_at": frame["datetime"].to_numpy(),
        }
    )


# Shorter box side in frame pixels below which a box is a padding phantom. The phantoms in the
# Phase 2 test crops are all under 4 px; the smallest real box is far above 16.
PHANTOM_PX = 8


def phantom_boxes(frame_paths: pd.Series, boxes: pd.Series) -> np.ndarray:
    from PIL import Image

    out = []
    for f, b in zip(frame_paths, boxes):
        if not isinstance(f, str) or not isinstance(b, str):
            out.append(False)
            continue
        x1, y1, x2, y2 = json.loads(b)
        with Image.open(f) as im:  # reads the header only
            w, h = im.size
        out.append(min((x2 - x1) * w, (y2 - y1) * h) < PHANTOM_PX)
    return np.array(out)


def species_frames(cfg: dict, seq_keys: pd.Series) -> pd.DataFrame:
    cols = ["image_id", "seq_key", "frame_num", "datetime", "local_path"]
    m = pd.read_csv(cfg["manifest"], usecols=cols, dtype={"image_id": str})
    m = m[m["seq_key"].isin(set(seq_keys))].sort_values(["seq_key", "frame_num", "datetime"])
    return pd.DataFrame(
        {
            "image_id": m["image_id"],
            "seq_key": m["seq_key"],
            "frame_num": m["frame_num"],
            "captured_at": m["datetime"],
            "frame_path": [Path(f).as_posix() for f in m["local_path"]],
        }
    )


def reid_tables(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    crops = pd.read_csv(cfg["crops"])
    test = crops[crops["split"] == "test"].reset_index(drop=True)
    emb = np.load(cfg["embeddings"])["x"]
    if len(emb) != len(test):
        raise ValueError(f"{cfg['embeddings']} has {len(emb)} rows, expected {len(test)} test crops")

    in_cat = split_catalogue(test, cfg["known_frac"], cfg["seed"]).to_numpy()
    cat, items = test[in_cat], test[~in_cat].reset_index(drop=True)
    lists = shortlists(
        emb[~in_cat],
        items["viewpoint"],
        emb[in_cat],
        cat["name"],
        cat["viewpoint"],
        cat["crop_id"].astype(str),
        cfg["shortlist"],
    )
    known_flanks = set(zip(cat["name"], cat["viewpoint"]))
    best = np.array([s[0]["sim"] if s else np.nan for s in lists])
    second = np.array([s[1]["sim"] if len(s) > 1 else np.nan for s in lists])
    items_df = pd.DataFrame(
        {
            "id": np.arange(1, len(items) + 1),
            "crop_id": items["crop_id"].astype(str),
            "viewpoint": items["viewpoint"],
            "encounter": items["encounter"],
            "best_sim": best,
            "margin": best - second,
            "suggest_new": (np.nan_to_num(best, nan=-1) < cfg["new_threshold"]).astype(int),
            "true_leopard": items["name"],
            # "New" means no catalogue photo of this leopard's flank: nothing to match it to.
            "truth_is_new": [int((n, v) not in known_flanks) for n, v in zip(items["name"], items["viewpoint"])],
        }
    )
    cands = pd.DataFrame([{"item_id": i + 1, **c} for i, s in enumerate(lists) for c in s])
    catalogue = pd.DataFrame(
        {
            "crop_id": cat["crop_id"].astype(str),
            "leopard": cat["name"],
            "viewpoint": cat["viewpoint"],
            "encounter": cat["encounter"],
        }
    )
    return catalogue, items_df, cands


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase5.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    species = species_rows(cfg["species"])
    species["phantom"] = phantom_boxes(species["frame_path"], species["bbox"]).astype(int)
    frames = species_frames(cfg["species"], species["seq_key"])
    catalogue, items, cands = reid_tables(cfg["reid"])
    rng = np.random.default_rng(cfg["audit_seed"])
    species["audit_rank"] = pd.array(audit_ranks(len(species), cfg["audit_frac"], rng), dtype="Int64")
    items["audit_rank"] = pd.array(audit_ranks(len(items), cfg["audit_frac"], rng), dtype="Int64")

    db = Path(cfg["db"])
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    with con:
        con.executescript(SCHEMA)
        species.to_sql("species_items", con, if_exists="append", index=False)
        frames.to_sql("species_frames", con, if_exists="append", index=False)
        catalogue.to_sql("reid_catalogue", con, if_exists="append", index=False)
        items.to_sql("reid_items", con, if_exists="append", index=False)
        cands.to_sql("reid_candidates", con, if_exists="append", index=False)
        meta = {
            "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "species_model": "SpeciesNet features + logistic regression (Phase 2 probe)",
            "reid_model": cfg["reid"]["embeddings"],
            "new_threshold": cfg["reid"]["new_threshold"],
            "species_images": cfg["species"]["images"],
            "reid_images": cfg["reid"]["images"],
        }
        con.executemany("INSERT INTO meta VALUES (?, ?)", [(k, str(v)) for k, v in meta.items()])
    con.close()

    sp_acc = (species["pred"] == species["true_label"]).mean()
    top1 = cands[cands["rank"] == 1].set_index("item_id")["leopard"]
    known = items[items["truth_is_new"] == 0]
    print(
        f"species: {len(species):,} crops, model accuracy {sp_acc:.1%}, frames found {species['frame_path'].notna().mean():.0%}, "
        f"{species['phantom'].sum()} padding phantoms flagged, "
        f"{len(frames):,} burst frames"
    )
    print(
        f"re-id: catalogue {len(catalogue)} crops of {catalogue['leopard'].nunique()} leopards; "
        f"{len(items)} to review ({items['truth_is_new'].mean():.0%} truly new)"
    )
    print(
        f"  known items with correct top-1: {(known['id'].map(top1) == known['true_leopard']).mean():.1%}; "
        f"suggested new: {items['suggest_new'].mean():.0%}"
    )
    print(f"Wrote {db}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
