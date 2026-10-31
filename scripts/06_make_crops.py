"""Phase 2, step 1: cut detector boxes into labelled crops, split by camera.

    python scripts/06_make_crops.py --config configs/phase2.yaml \
        --manifest data/swg/manifest_phase2.csv \
        --detections outputs/phase2/detections.jsonl --out data/crops --zip

Writes:
  <out>/images/<crop_id>.jpg   one crop per animal box (top --max-per-image per frame)
  <out>/crops.csv              crop_id, image_id, seq_key, location, split, label,
                               train_ok, conf, bbox
  <out>/splits.csv             camera location -> train | val | test
  <out>.zip                    with --zip, the folder ready to upload to Kaggle

Crops that already exist are not rewritten, so a stopped run can be resumed.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from ghats.config import load_config
from ghats.crops import BACKGROUND, SPLITS, crop_box, is_padding_strip, label_boxes, split_locations


def load_animal_boxes(path: Path, max_per_image: int) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            animal = sorted((d for d in rec["detections"] if d["category"] == "animal"), key=lambda d: -d["conf"])
            for k, d in enumerate(animal[:max_per_image]):
                rows.append({"image_id": str(rec["image_id"]), "box": k, "conf": d["conf"], "bbox": d["bbox"]})
    return pd.DataFrame(rows).drop_duplicates(["image_id", "box"], keep="last")


def frame_size(path: str) -> tuple[int, int]:
    with Image.open(path) as im:  # reads the header only
        return im.size


def write_crops(path: str, crops: list[tuple[str, list[float]]], out_dir: Path, c: dict) -> int:
    """Cut every crop of one image. Returns how many failed."""
    todo = [(cid, bbox) for cid, bbox in crops if not (out_dir / f"{cid}.jpg").exists()]
    if not todo:
        return 0
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            for cid, bbox in todo:
                crop = im.crop(crop_box(bbox, im.width, im.height, c["pad"], c["min_side"]))
                crop.thumbnail((c["max_side"], c["max_side"]), Image.Resampling.LANCZOS)
                crop.save(out_dir / f"{cid}.jpg", quality=c["jpeg_quality"])
    except Exception as e:  # noqa: BLE001 - one bad JPEG should not stop the run
        tqdm.write(f"skip {path}: {e}")
        return len(todo)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase2.yaml")
    ap.add_argument("--manifest", default="data/swg/manifest_phase2.csv")
    ap.add_argument("--detections", default="outputs/phase2/detections.jsonl")
    ap.add_argument("--out", default="data/crops")
    ap.add_argument("--zip", action="store_true", help="Also write <out>.zip for upload")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    c = load_config(args.config)["crops"]
    manifest = pd.read_csv(args.manifest, dtype={"image_id": str})

    # Split whole cameras, using every event in the sample (not only events with boxes).
    events = manifest.drop_duplicates("seq_key")
    events = events.assign(label=events["seq_group"].where(events["seq_gt"] == "animal", BACKGROUND))
    splits = split_locations(events, c["val_frac"], c["test_frac"], seed=c.get("seed", 42))

    boxes = load_animal_boxes(Path(args.detections), c["max_per_image"])
    cols = ["image_id", "local_path", "seq_key", "location", "seq_gt", "seq_group"]
    boxes = boxes.merge(manifest[cols], on="image_id", how="inner")
    crops = label_boxes(boxes, c["min_conf"], c["train_min_conf"])
    crops["split"] = crops["location"].map(splits)
    crops["crop_id"] = crops["image_id"] + "_" + crops["box"].astype(str)
    # Drop padding phantoms only now, after the ids are set: box numbers are confidence ranks
    # within a frame, and renumbering would point real crops at a phantom's saved image.
    with ThreadPoolExecutor(args.threads) as pool:
        sizes = dict(zip(crops["local_path"].unique(), pool.map(frame_size, crops["local_path"].unique())))
    strip = [is_padding_strip(b, *sizes[p]) for b, p in zip(crops["bbox"], crops["local_path"])]
    phantom = pd.Series(strip, index=crops.index)
    print(f"dropped {phantom.sum():,} padding-strip boxes")
    crops = crops[~phantom]

    out = Path(args.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    by_image = crops.groupby("local_path")[["crop_id", "bbox"]].apply(lambda g: list(zip(g["crop_id"], g["bbox"])))
    with ThreadPoolExecutor(args.threads) as pool:
        jobs = [pool.submit(write_crops, p, cs, img_dir, c) for p, cs in by_image.items()]
        failed = sum(j.result() for j in tqdm(jobs, desc="images"))

    keep = ["crop_id", "image_id", "seq_key", "location", "split", "label", "train_ok", "conf", "bbox"]
    crops = crops[crops["crop_id"].map(lambda cid: (img_dir / f"{cid}.jpg").exists())]
    crops[keep].to_csv(out / "crops.csv", index=False)
    pd.Series(splits, name="split").rename_axis("location").to_csv(out / "splits.csv")

    table = crops[crops["train_ok"]].pivot_table(index="label", columns="split", values="crop_id", aggfunc="size")
    print(table.reindex(columns=list(SPLITS)).fillna(0).astype(int).to_string())
    print(f"{len(crops):,} crops ({crops['train_ok'].sum():,} trainable), {failed} failed, in {out}")

    if args.zip:
        zpath = out.with_suffix(".zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as zf:  # JPEGs do not compress further
            for p in [out / "crops.csv", out / "splits.csv", *sorted(img_dir.glob("*.jpg"))]:
                zf.write(p, p.relative_to(out))
        print(f"Wrote {zpath} ({zpath.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
