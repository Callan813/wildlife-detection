"""Phase 2, step 5: look at the animal events the classifier filter throws away.

    python scripts/10_miss_gallery.py --threshold 0.504

At an operating point from reports/phase2/event_eval.csv ("ours, best box") some animal
events have no box the classifier believes. This draws every frame of such val/test
events, one event per row, with all detector boxes (yellow, the best box by classifier
score in red), so you can tell label noise, tiny animals and detector misses apart.
Events where the classifier rejected a box come first; the csv lists every event.
Writes reports/phase2/missed_events.{jpg,csv}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from ghats.crops import BACKGROUND

TILE = (360, 270)
CAPTION = 18


def frame_tile(path: str, boxes: pd.DataFrame, best_crop: str | None, caption: str) -> Image.Image:
    try:
        im = Image.open(path).convert("RGB")
    except OSError:
        im = Image.new("RGB", TILE, "gray")
    w, h = im.size
    draw = ImageDraw.Draw(im)
    for r in boxes.itertuples():
        x1, y1, x2, y2 = r.bbox
        colour = (255, 40, 40) if r.crop_id == best_crop else (255, 210, 0)
        draw.rectangle([x1 * w, y1 * h, x2 * w, y2 * h], outline=colour, width=max(3, w // 250))
    im.thumbnail(TILE)
    canvas = Image.new("RGB", (TILE[0], TILE[1] + CAPTION), "black")
    canvas.paste(im, ((TILE[0] - im.width) // 2, 0))
    ImageDraw.Draw(canvas).text((4, TILE[1] + 3), caption[:58], fill="white")
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", default="outputs/phase2/classifier/predictions.csv")
    ap.add_argument("--crops", default="data/crops/crops.csv")
    ap.add_argument("--manifest", default="data/swg/manifest_phase2.csv")
    ap.add_argument("--splits", default="data/crops/splits.csv")
    ap.add_argument("--threshold", type=float, required=True, help="P(not background) cut, picked on val")
    ap.add_argument("--max-frames", type=int, default=5)
    ap.add_argument("--max-events", type=int, default=24, help="Rows in the picture")
    ap.add_argument("--out", default="reports/phase2")
    args = ap.parse_args()

    preds = pd.read_csv(args.predictions)
    bbox = pd.read_csv(args.crops, usecols=["crop_id", "bbox"])
    boxes = preds.merge(bbox, on="crop_id")
    boxes["bbox"] = boxes["bbox"].map(json.loads)
    boxes["score"] = 1.0 - boxes[f"p_{BACKGROUND}"]
    boxes["area"] = boxes["bbox"].map(lambda b: (b[2] - b[0]) * (b[3] - b[1]))

    frames = pd.read_csv(args.manifest, dtype={"image_id": str})
    split = pd.read_csv(args.splits).set_index("location")["split"]
    events = frames.drop_duplicates("seq_key")[["seq_key", "location", "seq_gt", "seq_group"]]
    events = events.assign(split=events["location"].map(split))
    events = events[events["split"].isin(["val", "test"]) & (events["seq_gt"] == "animal")]

    best = boxes.sort_values("score", ascending=False).drop_duplicates("seq_key").set_index("seq_key")
    events = events.join(best[["crop_id", "score", "conf", "area", "pred"]], on="seq_key")
    events["score"] = events["score"].fillna(0.0)
    missed = events[events["score"] < args.threshold].sort_values(["split", "score"])
    no_box = missed["crop_id"].isna()
    print(f"{len(missed)} of {len(events)} val/test animal events below {args.threshold}")
    print(
        f"{no_box.sum()} have no detector box (detector misses), {(~no_box).sum()} have boxes the classifier rejected"
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    missed.drop(columns="seq_gt").to_csv(out / "missed_events.csv", index=False)
    if missed.empty:
        return 0

    rows = []
    shown = missed.assign(no_box=no_box).sort_values(["no_box", "score"]).head(args.max_events)
    for ev in shown.itertuples():
        ev_frames = frames[frames["seq_key"] == ev.seq_key].sort_values("frame_num").head(args.max_frames)
        tiles = []
        for i, f in enumerate(ev_frames.itertuples()):
            caption = f"{ev.split} {ev.seq_group} p={ev.score:.2f}" if i == 0 else f"{ev.location} frame {f.frame_num}"
            tiles.append(frame_tile(f.local_path, boxes[boxes["image_id"] == f.image_id], ev.crop_id, caption))
        rows.append(tiles)

    tw, th = TILE[0], TILE[1] + CAPTION
    sheet = Image.new("RGB", (args.max_frames * tw, len(rows) * th), "black")
    for r, tiles in enumerate(rows):
        for c, t in enumerate(tiles):
            sheet.paste(t, (c * tw, r * th))
    sheet.save(out / "missed_events.jpg", quality=85)
    print(f"Wrote {out / 'missed_events.jpg'} and missed_events.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
