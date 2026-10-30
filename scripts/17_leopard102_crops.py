"""Phase 3 extra, step 2: cut LeopardID102 crops from MegaDetector boxes.

    python scripts/17_leopard102_crops.py --config configs/phase3.yaml

Needs the manifest (16_leopard102_manifest.py) and detections (02_run_detector.py).
Takes the top animal box per named image, padded and resized like the Leopard ID 2022
crops. Writes:
  <out>/images/<crop_id>.jpg
  <out>/crops.csv   crop_id, image_id, name, camera, time, pair_side, encounter, conf, bbox

There is no viewpoint column yet: the dataset has no flank labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from ghats.config import load_config
from ghats.reid import time_encounters


def top_animal_box(detections: list[dict]) -> tuple[float, list[float]] | None:
    boxes = [d for d in detections if d["category"] == "animal"]
    if not boxes:
        return None
    best = max(boxes, key=lambda d: d["conf"])
    return best["conf"], best["bbox"]


def write_crop(src: str, dst: Path, bbox: list[float], c: dict) -> bool:
    if dst.exists():
        return True
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            x1, y1, x2, y2 = (v * s for v, s in zip(bbox, (im.width, im.height, im.width, im.height)))
            px, py = (x2 - x1) * c["pad"], (y2 - y1) * c["pad"]
            box = (max(0, x1 - px), max(0, y1 - py), min(im.width, x2 + px), min(im.height, y2 + py))
            crop = im.crop(tuple(round(v) for v in box))
            crop.thumbnail((c["max_side"], c["max_side"]), Image.Resampling.LANCZOS)
            crop.save(dst, quality=c["jpeg_quality"])
        return True
    except Exception as e:  # noqa: BLE001 - one bad JPEG should not stop the run
        tqdm.write(f"skip {src}: {e}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase3.yaml")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config)
    c, lp = cfg["crops"], cfg["leopard102"]
    m = pd.read_csv(lp["manifest"], keep_default_na=False)
    m = m[m["name"] != ""].copy()

    top = {}
    with open(lp["detections"], encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            top[r["image_id"]] = top_animal_box(r["detections"])
    m["conf"] = m["image_id"].map(lambda i: top[i][0] if top.get(i) else 0.0)
    m["bbox"] = m["image_id"].map(lambda i: top[i][1] if top.get(i) else None)
    kept = m[m["conf"] >= lp["min_conf"]].copy()
    print(f"{len(m)} named images, {len(kept)} with an animal box >= {lp['min_conf']}")

    kept["encounter"] = time_encounters(kept["name"], kept["camera"], kept["time"], lp["encounter_gap_min"])
    kept["crop_id"] = [f"lp102_{i:04d}" for i in range(len(kept))]

    out = Path(lp["out"])
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.threads) as pool:
        jobs = [
            pool.submit(write_crop, r.local_path, img_dir / f"{r.crop_id}.jpg", r.bbox, c) for r in kept.itertuples()
        ]
        ok = [j.result() for j in tqdm(jobs, desc="crops")]
    kept = kept[ok]
    cols = ["crop_id", "image_id", "name", "camera", "time", "pair_side", "encounter", "conf", "bbox"]
    kept[cols].to_csv(out / "crops.csv", index=False)

    per_leo = kept.groupby("name")["encounter"].nunique()
    print(f"{len(kept)} crops, {kept['name'].nunique()} leopards, {kept['encounter'].nunique()} encounters")
    print(f"leopards seen in 2+ encounters (usable as queries): {(per_leo >= 2).sum()}")
    print(f"crops per encounter: median {kept.groupby('encounter').size().median():.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
