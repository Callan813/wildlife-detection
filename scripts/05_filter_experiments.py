"""Phase 1, step 5: can a simple rule on top of the detector reject more false alarms?

    python scripts/05_filter_experiments.py

Compares event scoring rules from ``ghats.filters`` against the max-score
baseline, using the saved detections (no detector run needed). For each rule it
reports the operating point at several recall targets and writes
``reports/phase1/filter_experiments.md`` and ``filter_experiments.csv``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from ghats.config import load_config
from ghats.filters import (
    background_suppressed_score,
    kth_frame_score,
    max_score,
    spatial_agreement_score,
)
from ghats.metrics import pick_operating_point, threshold_sweep

TARGETS = [0.98, 0.95, 0.90]
# Boxes below this confidence are too weak to mark a spot as "background".
BACKGROUND_MIN_CONF = 0.1


def load_boxes(path: Path) -> dict[str, list]:
    boxes = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            boxes[str(rec["image_id"])] = [
                (d["conf"], d["bbox"]) for d in rec["detections"] if d["category"] == "animal"
            ]
    return boxes


def event_frames(manifest: pd.DataFrame, boxes: dict[str, list]) -> pd.DataFrame:
    df = manifest[manifest["image_id"].isin(boxes)].copy()
    df["boxes"] = df["image_id"].map(boxes)
    return (
        df.groupby("seq_key")
        .agg(gt=("seq_gt", "first"), location=("location", "first"), frames=("boxes", list))
        .reset_index()
    )


def score_rules(events: pd.DataFrame) -> dict[str, pd.Series]:
    # Boxes each camera produced, per event, for the background rule. No labels are used.
    strong = {
        e.seq_key: [b for frame in e.frames for c, b in frame if c >= BACKGROUND_MIN_CONF] for e in events.itertuples()
    }
    by_location = events.groupby("location")["seq_key"].apply(list).to_dict()

    def others(e):
        return [b for k in by_location[e.location] if k != e.seq_key for b in strong[k]]

    rules = {
        "Max score (baseline)": lambda e: max_score(e.frames),
        "Burst: 2 frames agree": lambda e: kth_frame_score(e.frames, k=2),
        "Burst: 3 frames agree": lambda e: kth_frame_score(e.frames, k=3),
        "Burst: same box in 2 frames": lambda e: spatial_agreement_score(e.frames, k=2),
        "Burst: same box in 3 frames": lambda e: spatial_agreement_score(e.frames, k=3),
        "Camera background (IoU 0.5)": lambda e: background_suppressed_score(e.frames, others(e), min_iou=0.5),
        "Camera background (IoU 0.7)": lambda e: background_suppressed_score(e.frames, others(e), min_iou=0.7),
    }
    return {name: pd.Series([fn(e) for e in events.itertuples()], index=events.index) for name, fn in rules.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--manifest", default="data/swg/manifest.csv")
    ap.add_argument("--detections", default="outputs/phase1/detections.jsonl")
    ap.add_argument("--out", default="reports/phase1")
    args = ap.parse_args()

    thresholds = load_config(args.config)["evaluation"]["thresholds"]
    manifest = pd.read_csv(args.manifest, dtype={"image_id": str})
    events = event_frames(manifest, load_boxes(Path(args.detections)))
    is_animal = events["gt"] == "animal"
    n_frames = events["frames"].map(len)

    rows = []
    for name, scores in score_rules(events).items():
        sweep = threshold_sweep(scores, is_animal, thresholds)
        for target in TARGETS:
            op = pick_operating_point(sweep, target)
            rows.append(
                {
                    "rule": name,
                    "target_recall": target,
                    "threshold": op["threshold"],
                    "recall": op["recall"],
                    "empty_removed": op["empty_removed"],
                    "meets_target": bool(op["recall"] >= target),
                }
            )
    table = pd.DataFrame(rows)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "filter_experiments.csv", index=False)

    per_loc = events.groupby("location").size()
    lines = [
        "# Phase 1: filter experiments",
        "",
        (
            f"{len(events):,} events. Frames per event: median {n_frames.median():.0f}, "
            f"{(n_frames == 1).mean():.0%} are a single frame. "
            f"Events per camera: median {per_loc.median():.0f}, max {per_loc.max()}."
        ),
        "",
        "| Rule | Target recall | Threshold | Recall | Empty events removed |",
        "|---|---|---|---|---|",
    ]
    for r in table.itertuples():
        flag = "" if r.meets_target else " (misses target)"
        lines.append(
            f"| {r.rule} | {r.target_recall:.0%} | {r.threshold:.2f} | {r.recall:.1%}{flag} | {r.empty_removed:.1%} |"
        )
    (out / "filter_experiments.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
