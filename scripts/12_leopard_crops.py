"""Phase 3, step 1: cut leopard crops from Leopard ID 2022 and split by individual.

    python scripts/12_leopard_crops.py --config configs/phase3.yaml

Download and unpack the dataset first (8 GB, see `dataset.url` in the config) into
data/leopard/. Writes:
  <out>/images/<annotation id>.jpg   one crop per left or right flank view
  <out>/crops.csv                    crop_id, image_id, name, viewpoint, encounter, split

Encounters are computed over every annotation (all viewpoints) before filtering, so a
front view in the middle of a sighting does not split it in two. Crops that already
exist are not rewritten.
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
from ghats.reid import assign_encounters, split_individuals


def write_crop(src: Path, dst: Path, bbox: list[float], c: dict) -> bool:
    if dst.exists():
        return True
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            x, y, w, h = bbox
            px, py = w * c["pad"], h * c["pad"]
            box = (max(0, x - px), max(0, y - py), min(im.width, x + w + px), min(im.height, y + h + py))
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
    d, c, s = cfg["dataset"], cfg["crops"], cfg["split"]
    root = Path(d["root"])
    coco = json.loads((root / d["annotations"]).read_text(encoding="utf-8"))
    files = {im["id"]: im["file_name"] for im in coco["images"]}
    ann = pd.DataFrame(
        [
            {
                "crop_id": a["id"],
                "image_id": a["image_id"],
                "name": a["name"],
                "viewpoint": a["viewpoint"],
                "bbox": a["bbox"],
            }
            for a in coco["annotations"]
        ]
    )
    ann["encounter"] = assign_encounters(ann["image_id"], ann["name"])
    print(f"{len(ann):,} annotations, {ann['name'].nunique()} leopards, {ann['encounter'].nunique():,} encounters")

    crops = ann[ann["viewpoint"].isin(c["viewpoints"])].copy()
    split = split_individuals(crops["name"], s["val_frac"], s["test_frac"], s["seed"])
    crops["split"] = crops["name"].map(split)

    out = Path(c["out"])
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    src_dir = root / d["images"]
    with ThreadPoolExecutor(args.threads) as pool:
        jobs = [
            pool.submit(write_crop, src_dir / files[r.image_id], img_dir / f"{r.crop_id}.jpg", r.bbox, c)
            for r in crops.itertuples()
        ]
        ok = [j.result() for j in tqdm(jobs, desc="crops")]
    crops = crops[ok]
    crops.drop(columns="bbox").to_csv(out / "crops.csv", index=False)

    table = crops.groupby("split").agg(
        crops=("crop_id", "size"), leopards=("name", "nunique"), encounters=("encounter", "nunique")
    )
    print(table.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
