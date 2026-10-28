"""Phase 3, step 2: how well do off-the-shelf features re-identify leopards, with no training?

    python scripts/13_reid_baseline.py --config configs/phase3.yaml

Every val/test crop from scripts/12_leopard_crops.py is embedded by each model in the
config's `reid.models` (DINOv2 at two sizes, and SpeciesNet's backbone from Phase 2).
Each crop is then a query against all other crops of its split: top-k accuracy and mAP,
with same-encounter crops removed and only same-flank matches counted (ghats.reid).

Also reports how often a correct top-1 match is an image within 10 ids of the query: a
high share would mean the encounter proxy splits sightings and the score is flattered.
Embeddings are cached in outputs/phase3/embeddings/<model>.npz. Writes
reports/phase3/reid_baseline.{md,csv}.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ghats.config import load_config
from ghats.reid import l2_normalise, retrieval_metrics

NEARBY = 10  # image ids


class AspectCrops:
    """Crops resized so the longest side is ``size``, both sides a multiple of 14 (ViT patch)."""

    def __init__(self, paths: list[Path], size: int, mean, std):
        from torchvision import transforms as T

        self.paths, self.size = paths, size
        self.norm = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB")
            scale = self.size / max(im.size)
            w, h = (max(14, round(v * scale / 14) * 14) for v in im.size)
            return self.norm(im.resize((w, h), Image.Resampling.BICUBIC))


def embed_timm(paths: list[Path], name: str, size: int, device: str) -> np.ndarray:
    import timm
    import torch

    model = timm.create_model(name, pretrained=True, num_classes=0, dynamic_img_size=True).to(device).eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    ds = AspectCrops(paths, size, cfg["mean"], cfg["std"])
    out = []
    with torch.no_grad(), torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
        for i in range(len(ds)):  # one at a time: crops keep their own aspect ratio
            out.append(model(ds[i][None].to(device)).float().cpu().numpy())
    return np.concatenate(out)


def embed_speciesnet(paths: list[Path], cfg_phase2: dict, device: str) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader

    from ghats.baselines import TightCrops, speciesnet_backbone, speciesnet_preprocess

    b = cfg_phase2["baselines"]["speciesnet"]
    model, feats = speciesnet_backbone(Path(b["dir"]) / b["weights"], device)
    whole = TightCrops([str(p) for p in paths], [[0.0, 0.0, 1.0, 1.0]] * len(paths), speciesnet_preprocess)

    with torch.no_grad():
        for x in DataLoader(whole, batch_size=16, num_workers=4):
            model(x.to(device))
    return np.concatenate(feats)


def nearby_share(emb: np.ndarray, crops: pd.DataFrame) -> float:
    """Of correct top-1 matches (same rules as retrieval_metrics), the share within NEARBY image ids."""
    ids, enc, view = crops["name"].to_numpy(), crops["encounter"].to_numpy(), crops["viewpoint"].to_numpy()
    img = crops["image_id"].to_numpy()
    sim = emb @ emb.T
    near = []
    for q in range(len(ids)):
        keep = (enc != enc[q]) & (view == view[q])
        keep[q] = False
        if not (keep & (ids == ids[q])).any():
            continue
        cand = np.flatnonzero(keep)
        top = cand[np.argmax(sim[q, cand])]
        if ids[top] == ids[q]:
            near.append(abs(int(img[top]) - int(img[q])) <= NEARBY)
    return float(np.mean(near)) if near else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase3.yaml")
    ap.add_argument("--phase2-config", default="configs/phase2.yaml", help="For the SpeciesNet weights")
    ap.add_argument("--cache", default="outputs/phase3/embeddings")
    ap.add_argument("--reports", default="reports/phase3")
    args = ap.parse_args()

    import torch

    cfg = load_config(args.config)
    r = cfg["reid"]
    out_dir = Path(cfg["crops"]["out"])
    crops = pd.read_csv(out_dir / "crops.csv")
    crops = crops[crops["split"].isin(["val", "test"])].reset_index(drop=True)
    paths = [out_dir / "images" / f"{c}.jpg" for c in crops["crop_id"]]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, m in r["models"].items():
        f = cache / f"{name}.npz"
        if f.exists() and list(np.load(f)["crop_id"]) == crops["crop_id"].tolist():
            emb = np.load(f)["x"]
        else:
            print(f"Embedding {len(paths):,} crops with {name} on {device}")
            if m["kind"] == "timm":
                emb = embed_timm(paths, m["name"], m["size"], device)
            else:
                emb = embed_speciesnet(paths, load_config(args.phase2_config), device)
            np.savez(f, x=emb, crop_id=crops["crop_id"].to_numpy())
        emb = l2_normalise(emb)
        for split in ("val", "test"):
            s = (crops["split"] == split).to_numpy()
            sub = crops[s]
            met = retrieval_metrics(emb[s], sub["name"], sub["encounter"], sub["viewpoint"], r["ks"])
            rows.append({"model": name, "split": split, **met, "top1_from_nearby_images": nearby_share(emb[s], sub)})
            print(
                f"{name:12s} {split:5s} "
                + "  ".join(f"{k} {v:.3f}" for k, v in rows[-1].items() if k.startswith(("top", "mAP")))
            )

    table = pd.DataFrame(rows)
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    table.to_csv(reports / "reid_baseline.csv", index=False)
    test = crops[crops["split"] == "test"]
    t = table[table["split"] == "test"].set_index("model")
    cols = [f"top{k}_per_id" for k in r["ks"]] + ["mAP", "top1"]
    heads = [f"Top-{k}" for k in r["ks"]] + ["mAP", "Top-1 per query"]
    v = table[table["split"] == "val"].set_index("model")
    lines = [
        "# Phase 3: leopard re-identification with off-the-shelf features (no training)",
        "",
        (
            f"Test: {len(test):,} left/right flank crops of {test['name'].nunique()} leopards never used for "
            f"training, in {test['encounter'].nunique()} encounters. Each crop is a query against all other test "
            "crops; crops from its own encounter are removed, and only the same flank counts as a match."
        ),
        "",
        (
            "Top-k is averaged per leopard (each individual's hit rate, then the mean), so the few leopards "
            "photographed 100+ times don't dominate; the per-query top-1 is shown for comparison."
        ),
        "",
        "| Model | " + " | ".join(heads) + f" | Val top-1 | Queries | Correct top-1 within {NEARBY} image ids |",
        "|---|" + "---|" * (len(cols) + 3),
        *(
            f"| {m} | " + " | ".join(f"{row[c]:.1%}" for c in cols) + f" | {v.loc[m, 'top1_per_id']:.1%} | "
            f"{int(row['queries'])} | {row['top1_from_nearby_images']:.0%} |"
            for m, row in t.iterrows()
        ),
        "",
        "All numbers for val and test, per query and per leopard, are in reid_baseline.csv.",
    ]
    (reports / "reid_baseline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
