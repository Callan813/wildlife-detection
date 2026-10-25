"""Phase 1, step 2: run MegaDetector V6 over every image in the manifest.

    python scripts/02_run_detector.py --manifest data/swg/manifest.csv \
        --out outputs/phase1/detections.jsonl

Writes one JSON line per image: {"image_id", "detections": [...], "ms"}.
Already-processed images are skipped, so you can stop and resume.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ghats.config import load_config
from ghats.detector import MegaDetector


def already_done(path: Path) -> set[str]:
    done: set[str] = set()
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(str(json.loads(line)["image_id"]))
                except (json.JSONDecodeError, KeyError):
                    continue  # a half-written last line from an interrupted run
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--manifest", default="data/swg/manifest.csv")
    ap.add_argument("--out", default="outputs/phase1/detections.jsonl")
    ap.add_argument("--limit", type=int, help="Only process the first N images (quick smoke test)")
    args = ap.parse_args()

    cfg = load_config(args.config)["detector"]
    manifest = pd.read_csv(args.manifest, dtype={"image_id": str})
    manifest = manifest[manifest["local_path"].map(lambda p: Path(p).exists())]
    if args.limit:
        manifest = manifest.head(args.limit)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = already_done(out)
    todo = manifest[~manifest["image_id"].isin(done)]
    print(f"{len(manifest):,} images on disk, {len(done):,} done, {len(todo):,} to run")
    if todo.empty:
        return 0

    det = MegaDetector(cfg["class"], cfg["version"], cfg.get("device", "auto"), cfg.get("conf_floor", 0.01))
    print(f"Model: {det.name} on {det.device}")

    failures = 0
    with open(out, "a", encoding="utf-8") as f:
        for row in tqdm(todo.itertuples(), total=len(todo)):
            try:
                dets, ms = det.detect(row.local_path)
            except Exception as e:  # noqa: BLE001 - a corrupt JPEG should not kill a 2-hour run
                failures += 1
                tqdm.write(f"skip {row.local_path}: {e}")
                continue
            rec = {
                "image_id": row.image_id,
                "model": det.name,
                "ms": round(ms, 2),
                "detections": [d.__dict__ for d in dets],
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
    print(f"Wrote {out} ({failures} failures)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
