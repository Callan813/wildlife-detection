"""Phase 3 extra, step 3: label LeopardID102 crops as left or right flank.

    python scripts/18_flank_classifier.py --config configs/phase3.yaml

LeopardID102 has no viewpoint labels, but re-ID only compares a flank with the same
flank. Leopard ID 2022 labels every crop, so a classifier is trained there and applied
to LeopardID102:

1. Frozen DINOv2-S features (the same backbone as the re-ID model) for both datasets,
   cached in outputs/phase3/flank/features.npz.
2. Logistic regression on the Leopard ID 2022 train leopards; C picked on val leopards,
   accuracy reported on test leopards.
3. Predict LeopardID102. There are no labels to score against, so:
   - A grid of random crops with their predicted side, checked by eye
     (reports/phase3/flank_check.jpg, gitignored: LeopardID102 is NC-SA). This is the real accuracy check.
   - a/b frames: files ending a and b with the same camera and second are two shots
     of the same moment from the same side (I first assumed facing cameras; looking
     at them showed otherwise). They should get the same side. Only a consistency
     check: many are near-identical, so agreement is expected.

Adds viewpoint and p_right columns to the LeopardID102 crops.csv. Writes
reports/phase3/flank.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.linear_model import LogisticRegression

from ghats.config import load_config

BACKBONE = "vit_small_patch14_dinov2.lvd142m"


def features(paths: list[Path], size: tuple[int, int], device: str) -> np.ndarray:
    import timm
    import torch
    from torch.utils.data import DataLoader

    from ghats.reid_model import Crops, transforms

    model = timm.create_model(BACKBONE, pretrained=True, num_classes=0, dynamic_img_size=True).to(device).eval()
    pc = model.pretrained_cfg
    loader = DataLoader(
        Crops(paths, [0] * len(paths), transforms(size, pc["mean"], pc["std"], 0, train=False)),
        batch_size=32,
        num_workers=4,
    )
    out = []
    with torch.no_grad():
        for x, _ in loader:
            with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                # CLS token plus mean patch token. DINOv2 was pretrained with flips, so the
                # CLS token alone is partly blind to which way the animal faces; the
                # patch tokens keep their position.
                tokens = model.forward_features(x.to(device))
                n_patch_w = size[1] // 14
                patches = tokens[:, model.num_prefix_tokens :].float()
                patches = patches.reshape(len(x), -1, n_patch_w, patches.shape[-1])
                left_half = patches[:, :, : n_patch_w // 2].mean((1, 2))
                right_half = patches[:, :, n_patch_w // 2 :].mean((1, 2))
                f = torch.cat([tokens[:, 0].float(), left_half - right_half], dim=1)
            out.append(f.cpu().numpy())
    return np.concatenate(out)


def pair_check(lp: pd.DataFrame) -> pd.DataFrame:
    """Same leopard, same camera, same second, one 'a' and one 'b' frame: sides should match."""
    key = ["name", "camera", "time"]
    ab = lp[lp["pair_side"].isin(["a", "b"])].drop_duplicates(key + ["pair_side"], keep=False)  # 1 frame per side
    a, b = (ab[ab["pair_side"] == s][key + ["viewpoint", "p_right"]] for s in ("a", "b"))
    pairs = a.merge(b, on=key, suffixes=("_a", "_b"))
    pairs["same"] = pairs["viewpoint_a"] == pairs["viewpoint_b"]
    return pairs


def markdown(df: pd.DataFrame, fmt) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.to_numpy()]
    return "\n".join([head, rule, *body])


def check_grid(lp: pd.DataFrame, img_dir: Path, out: Path, n: int = 40, seed: int = 0) -> pd.DataFrame:
    sample = lp.sample(n, random_state=seed).reset_index(drop=True)
    w, h, cols = 240, 160, 8
    grid = Image.new("RGB", (w * cols, h * (n // cols)), "white")
    draw = ImageDraw.Draw(grid)
    for k, r in sample.iterrows():
        with Image.open(img_dir / f"{r.crop_id}.jpg") as im:
            im.thumbnail((w - 4, h - 22))
            x, y = (k % cols) * w, (k // cols) * h
            grid.paste(im, (x, y + 20))
        draw.text((x + 2, y + 2), f"{k}: {r.viewpoint.upper()} ({max(r.p_right, 1 - r.p_right):.2f})", fill="black")
    grid.save(out, quality=88)
    return sample[["crop_id", "name", "viewpoint", "p_right"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase3.yaml")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    size = tuple(cfg["train"]["size"])
    lid_dir = Path(cfg["crops"]["out"])
    lp_dir = Path(cfg["leopard102"]["out"])
    lid = pd.read_csv(lid_dir / "crops.csv")
    lp = pd.read_csv(lp_dir / "crops.csv", keep_default_na=False)

    cache = Path("outputs/phase3/flank/features.npz")
    if cache.exists():
        z = np.load(cache)
        x_lid, x_lp = z["lid"], z["lp"]
    else:
        cache.parent.mkdir(parents=True, exist_ok=True)
        x_lid = features([lid_dir / "images" / f"{c}.jpg" for c in lid["crop_id"]], size, args.device)
        x_lp = features([lp_dir / "images" / f"{c}.jpg" for c in lp["crop_id"]], size, args.device)
        np.savez(cache, lid=x_lid, lp=x_lp)

    y = (lid["viewpoint"] == "right").to_numpy()
    part = {s: (lid["split"] == s).to_numpy() for s in ("train", "val", "test")}
    dim = x_lid.shape[1] // 2
    feature_sets = {"cls only": slice(0, dim), "cls + left-right patch difference": slice(0, 2 * dim)}

    rows, best = [], None
    for fname, cols in feature_sets.items():
        for c in (0.01, 0.1, 1.0):
            clf = LogisticRegression(C=c, max_iter=5000).fit(x_lid[part["train"], cols], y[part["train"]])
            acc = {s: (clf.predict(x_lid[part[s], cols]) == y[part[s]]).mean() for s in ("val", "test")}
            rows.append({"features": fname, "C": c, "val_acc": acc["val"], "test_acc": acc["test"]})
            if best is None or acc["val"] > best[0]:
                best = (acc["val"], fname, c, cols, clf)
    sweep = pd.DataFrame(rows)
    print(sweep.to_string(index=False, float_format="%.3f"))
    _, fname, c, cols, clf = best
    print(f"chosen on val: {fname}, C={c}")

    lp["p_right"] = clf.predict_proba(x_lp[:, cols])[:, 1].round(4)
    lp["viewpoint"] = np.where(lp["p_right"] >= 0.5, "right", "left")
    lp.to_csv(lp_dir / "crops.csv", index=False)

    pairs = pair_check(lp)
    unsure = (lp["p_right"].sub(0.5).abs() < 0.3).mean()
    report_dir = Path("reports/phase3")
    sample = check_grid(lp, lp_dir / "images", report_dir / "flank_check.jpg")
    sample.to_csv(report_dir / "flank_check.csv", index_label="grid_index")

    test_row = sweep[(sweep["features"] == fname) & (sweep["C"] == c)].iloc[0]
    lines = [
        "# Flank classifier: Leopard ID 2022 labels applied to LeopardID102",
        "",
        "Generated by `scripts/18_flank_classifier.py`. Frozen DINOv2-S features + logistic regression,",
        "trained on Leopard ID 2022 train leopards; C and features picked on val leopards.",
        "",
        "## Leopard ID 2022 (labelled)",
        "",
        markdown(sweep, lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)),
        "",
        (
            f"Chosen: **{fname}, C={c}**: test accuracy **{test_row['test_acc']:.1%}** "
            f"({part['test'].sum()} crops of unseen leopards)."
        ),
        "",
        "## LeopardID102 (no labels)",
        "",
        f"- {len(lp)} crops: {(lp['viewpoint'] == 'left').sum()} left, {(lp['viewpoint'] == 'right').sum()} right.",
        f"- {unsure:.1%} of predictions have p between 0.2 and 0.8.",
        (
            f"- a/b consistency: {len(pairs)} a/b frame pairs (same leopard, camera and second) get the same "
            f"side in {pairs['same'].mean():.1%}. Many pairs are near-identical, so this is a sanity check only."
        ),
        (
            "- Visual check: `flank_check.jpg` (40 random crops, predicted side and confidence), "
            "indices in `flank_check.csv`."
        ),
    ]
    (report_dir / "flank.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-5:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
