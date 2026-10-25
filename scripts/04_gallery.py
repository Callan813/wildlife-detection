"""Phase 1, step 4: picture grids of the detector's mistakes, for the README.

    python scripts/04_gallery.py --threshold 0.2

Makes two images in reports/phase1/:
  * missed_animals.jpg: animal events the detector scored below the threshold
  * false_alarms.jpg:   empty events it scored highest (wind, leaves, sun glare)
Looking at these is how you find out *why* it fails, which is the part worth
writing up.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

TILE = (320, 240)


def load_boxes(path: Path) -> dict[str, list[dict]]:
    boxes: dict[str, list[dict]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            boxes[str(rec["image_id"])] = rec["detections"]
    return boxes


def tile(path: str, dets: list[dict], caption: str, min_conf: float) -> Image.Image:
    try:
        im = Image.open(path).convert("RGB")
    except OSError:
        im = Image.new("RGB", TILE, "gray")
    w, h = im.size
    draw = ImageDraw.Draw(im)
    for d in dets:
        if d["category"] != "animal" or d["conf"] < min_conf:
            continue
        x1, y1, x2, y2 = d["bbox"]
        draw.rectangle([x1 * w, y1 * h, x2 * w, y2 * h], outline=(255, 210, 0), width=max(2, w // 300))
    im.thumbnail(TILE)
    canvas = Image.new("RGB", (TILE[0], TILE[1] + 18), "black")
    canvas.paste(im, ((TILE[0] - im.width) // 2, 0))
    ImageDraw.Draw(canvas).text((4, TILE[1] + 3), caption[:52], fill="white")
    return canvas


def grid(tiles: list[Image.Image], cols: int = 4) -> Image.Image:
    rows = max(1, (len(tiles) + cols - 1) // cols)
    tw, th = TILE[0], TILE[1] + 18
    out = Image.new("RGB", (cols * tw, rows * th), "black")
    for i, t in enumerate(tiles):
        out.paste(t, ((i % cols) * tw, (i // cols) * th))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default="reports/phase1/frame_scores.csv")
    ap.add_argument("--detections", default="outputs/phase1/detections.jsonl")
    ap.add_argument("--out", default="reports/phase1")
    ap.add_argument("--threshold", type=float, required=True, help="Use the operating point from summary.md")
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()

    frames = pd.read_csv(args.scores, dtype={"image_id": str})
    boxes = load_boxes(Path(args.detections))
    # Best-scoring frame per event represents that event.
    best = frames.sort_values("score", ascending=False).drop_duplicates("seq_key")

    missed = best[(best["seq_gt"] == "animal") & (best["score"] < args.threshold)].head(args.n)
    false_alarms = best[(best["seq_gt"] == "empty") & (best["score"] >= args.threshold)].head(args.n)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, df, label in [
        ("missed_animals.jpg", missed, "missed"),
        ("false_alarms.jpg", false_alarms, "false alarm"),
    ]:
        if df.empty:
            print(f"No {label} cases at threshold {args.threshold}")
            continue
        tiles = [
            tile(
                r.local_path,
                boxes.get(r.image_id, []),
                f"{label}: {r.seq_group if pd.notna(r.seq_group) else 'empty'} ({r.score:.2f})",
                0.05,
            )
            for r in df.itertuples()
        ]
        grid(tiles).save(out / name, quality=88)
        print(f"Wrote {out / name} ({len(tiles)} examples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
