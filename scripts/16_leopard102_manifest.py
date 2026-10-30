"""Phase 3 extra, step 1: list LeopardID102 images for the detector.

    python scripts/16_leopard102_manifest.py

LeopardID102 (Zimbabwe camera-trap survey, CC BY-NC-SA 4.0) comes as one folder
per leopard plus No_ID for unidentified ones, with full frames and no boxes.
Download it from https://huggingface.co/datasets/WildCAT-Datasets/LeopardID102
and unzip into data/leopard102/. Writes data/leopard102/manifest.csv:
  image_id, local_path, name (empty for No_ID), camera, time, pair_side

Then run the detector on it:
    python scripts/02_run_detector.py --manifest data/leopard102/manifest.csv \
        --out outputs/phase3/leopard102/detections.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ghats.reid import parse_spotted_filename


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/leopard102/LeopardID102")
    ap.add_argument("--out", default="data/leopard102/manifest.csv")
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    # No_ID also holds a few cub subfolders (LpC*); they have no identity either.
    for path in sorted(root.rglob("*.JPG")):
        folder = path.relative_to(root).parts[0]
        rows.append(
            {
                "image_id": path.relative_to(root).with_suffix("").as_posix(),
                "local_path": path.as_posix(),
                "name": "" if folder == "No_ID" else folder,
                **parse_spotted_filename(path.stem),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    named = df[df["name"] != ""]
    print(f"{len(df)} images: {len(named)} of {named['name'].nunique()} leopards, {len(df) - len(named)} No_ID")
    print(f"{df['camera'].nunique()} cameras, {df['time'].min()} to {df['time'].max()}")
    print(f"paired a/b frames: {(df['pair_side'] != '').sum()}")
    per = named["name"].value_counts()
    print(f"images per leopard: median {per.median():.0f}, max {per.max()}, leopards with 1 image: {(per == 1).sum()}")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
