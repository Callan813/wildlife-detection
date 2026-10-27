"""Phase 1, step 2: run MegaDetector V6 over every image in the manifest.

    python scripts/02_run_detector.py --manifest data/swg/manifest.csv \
        --out outputs/phase1/detections.jsonl

Writes one JSON line per image: {"image_id", "detections": [...], "ms"}.
Already-processed images are skipped, so you can stop and resume.
With MegaDetectorV6MIT, images are decoded by --workers processes and run in
batches of --batch-size on the GPU; the "ms" field is then batch time / batch size.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ghats.config import load_config
from ghats.detector import MegaDetector, load_image


class ImageList:
    """Map-style dataset for torch's DataLoader. A bad image comes back with its exception."""

    def __init__(self, rows: list[tuple[str, str]], transform):
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        image_id, path = self.rows[i]
        try:
            return image_id, path, load_image(path, self.transform)
        except Exception as e:  # noqa: BLE001 - a corrupt JPEG should not kill a 2-hour run
            return image_id, path, e


def keep_items(batch):
    return batch  # stack on the main side, after dropping failed images


def record(image_id: str, model: str, ms: float, dets) -> str:
    return json.dumps(
        {"image_id": image_id, "model": model, "ms": round(ms, 2), "detections": [d.__dict__ for d in dets]}
    )


def run_batched(det: MegaDetector, todo: pd.DataFrame, f, batch_size: int, workers: int) -> int:
    import time

    import torch
    from torch.utils.data import DataLoader

    rows = list(zip(todo["image_id"], todo["local_path"]))
    loader = DataLoader(
        ImageList(rows, det.transform),
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=keep_items,
        pin_memory=det.device == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )
    failures = 0
    with tqdm(total=len(rows), unit="img") as bar:
        for batch in loader:
            good = [b for b in batch if not isinstance(b[2], Exception)]
            for image_id, path, err in batch:
                if isinstance(err, Exception):
                    failures += 1
                    tqdm.write(f"skip {path}: {err}")
            if good:
                start = time.perf_counter()
                images = torch.stack([g[2][0] for g in good])
                revs = torch.stack([g[2][1] for g in good])
                results = det.detect_batch(images, revs, [g[2][2] for g in good])
                ms = (time.perf_counter() - start) * 1000.0 / len(good)
                f.write("".join(record(g[0], det.name, ms, r) + "\n" for g, r in zip(good, results)))
                f.flush()
            bar.update(len(batch))
    return failures


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
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4, help="Image decoding processes; each takes ~0.8 GB RAM")
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
        if det.batched:
            failures = run_batched(det, todo, f, args.batch_size, args.workers)
        else:
            for row in tqdm(todo.itertuples(), total=len(todo)):
                try:
                    dets, ms = det.detect(row.local_path)
                except Exception as e:  # noqa: BLE001 - a corrupt JPEG should not kill a 2-hour run
                    failures += 1
                    tqdm.write(f"skip {row.local_path}: {e}")
                    continue
                f.write(record(row.image_id, det.name, ms, dets) + "\n")
                f.flush()
    print(f"Wrote {out} ({failures} failures)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
