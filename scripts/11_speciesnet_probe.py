"""Phase 2, step 6: SpeciesNet as a frozen feature extractor, with a new head for our groups.

    python scripts/11_speciesnet_probe.py
    python scripts/09_event_eval.py --ours ours=outputs/phase2/classifier/predictions.csv \
        probe=outputs/phase2/speciesnet_probe/predictions.csv

SpeciesNet off the shelf beat our ConvNeXt (scripts/08_run_baselines.py). Is that its
backbone, trained on 65M camera-trap images, or its head, which may have seen SWG? This
keeps the backbone frozen, takes its pooled 1,280-d features for every crop (tight boxes,
as for the baseline), and fits a logistic regression on our train cameras only. The Central
India model is built the same way (frozen SpeciesNet + linear head), just on Indian data.

Features are cached in <out>/features.npz, so changing the head does not rerun the network.
The regularisation C is picked on val macro F1, like the epoch in 07_train_classifier.py.
Writes <out>/predictions.csv (same columns as our classifier's), <out>/head.npz (scaler +
logistic regression as plain arrays, so a camera can apply the head with numpy alone) and
reports/phase2/classifier_{summary,test}_probe.*.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ghats.baselines import TightCrops, speciesnet_backbone, speciesnet_preprocess
from ghats.classify import class_weights, per_class_report
from ghats.config import load_config


def extract(crops: pd.DataFrame, weights: Path, batch_size: int, workers: int) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, feats = speciesnet_backbone(weights, device)
    loader = DataLoader(
        TightCrops(crops["local_path"].tolist(), [ast.literal_eval(s) for s in crops["bbox"]], speciesnet_preprocess),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=device == "cuda",
    )
    print(f"Extracting {len(crops):,} crops on {device}")
    with torch.no_grad():
        for i, x in enumerate(loader):
            model(x.to(device, non_blocking=True))
            if i % 100 == 0:
                print(f"  {min((i + 1) * batch_size, len(crops)):,}/{len(crops):,}", flush=True)
    return np.concatenate(feats)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase2.yaml")
    ap.add_argument("--crops", default="data/crops/crops.csv")
    ap.add_argument("--manifest", default="data/swg/manifest_phase2.csv")
    ap.add_argument("--ours", default="outputs/phase2/classifier/predictions.csv", help="For the group list")
    ap.add_argument("--out", default="outputs/phase2/speciesnet_probe")
    ap.add_argument("--reports", default="reports/phase2")
    ap.add_argument("--c", nargs="+", type=float, default=[0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    b = load_config(args.config)["baselines"]["speciesnet"]
    groups = sorted(pd.read_csv(args.ours, nrows=0).filter(regex="^p_").columns.str[2:])
    idx = {g: k for k, g in enumerate(groups)}

    crops = pd.read_csv(args.crops, dtype={"image_id": str})
    # Train on what 07 trains on; score every val/test crop, for the event-level eval.
    crops = crops[((crops["split"] == "train") & crops["train_ok"]) | crops["split"].isin(["val", "test"])]
    paths = pd.read_csv(args.manifest, dtype={"image_id": str}).set_index("image_id")["local_path"]
    crops = crops.assign(local_path=crops["image_id"].map(paths)).dropna(subset=["local_path"]).reset_index(drop=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "features.npz"
    if cache.exists() and set(np.load(cache, allow_pickle=True)["crop_id"]) >= set(crops["crop_id"]):
        z = np.load(cache, allow_pickle=True)
        X = pd.DataFrame(z["x"], index=z["crop_id"]).loc[crops["crop_id"]].to_numpy()
        print(f"Loaded {len(X):,} cached features")
    else:
        X = extract(crops, Path(b["dir"]) / b["weights"], args.batch_size, args.workers)
        np.savez(cache, x=X, crop_id=crops["crop_id"].to_numpy())

    y = crops["label"].map(idx).to_numpy()
    tr = (crops["split"] == "train").to_numpy()
    va = ((crops["split"] == "val") & crops["train_ok"]).to_numpy()
    te = ((crops["split"] == "test") & crops["train_ok"]).to_numpy()
    scaler = StandardScaler().fit(X[tr])
    Xs = scaler.transform(X)
    counts = np.bincount(y[tr], minlength=len(groups))
    sw = class_weights(counts)[y[tr]]  # same sqrt weighting as 07

    best = None
    for c in args.c:
        clf = LogisticRegression(C=c, max_iter=3000).fit(Xs[tr], y[tr], sample_weight=sw)
        f1 = per_class_report(y[va], clf.predict(Xs[va]), groups)["f1"].iloc[-1]
        print(f"C={c:<6} val macro F1 {f1:.3f}")
        if best is None or f1 > best[0]:
            best = (f1, c, clf)
    val_f1, c, clf = best

    probs = clf.predict_proba(Xs)
    pred = crops[["crop_id", "image_id", "seq_key", "location", "split", "label", "train_ok", "conf"]].copy()
    pred["pred"] = [groups[k] for k in probs.argmax(1)]
    pred["pred_prob"] = probs.max(1)
    pred = pd.concat([pred, pd.DataFrame(probs, columns=[f"p_{g}" for g in groups])], axis=1)
    pred[pred["split"] != "train"].to_csv(out / "predictions.csv", index=False)
    np.savez(
        out / "head.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
        coef=clf.coef_,
        intercept=clf.intercept_,
        classes=np.array(groups),
        c=c,
    )

    rep = per_class_report(y[te], probs[te].argmax(1), groups)
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    rep.to_csv(reports / "classifier_test_probe.csv", index=False)
    acc = float((probs[te].argmax(1) == y[te]).mean())
    body = rep.iloc[:-1].sort_values("f1")
    lines = [
        "# Phase 2: frozen SpeciesNet features + logistic regression",
        "",
        f"Head trained on {tr.sum():,} train-camera crops, C={c} picked on val (val macro F1 {val_f1:.3f}).",
        f"Test: {te.sum():,} crops from {crops.loc[te, 'location'].nunique()} cameras never seen in training.",
        "",
        f"**Test macro F1 {rep['f1'].iloc[-1]:.3f}**, accuracy {acc:.3f}.",
        "",
        "| Class | Test crops | Precision | Recall | F1 |",
        "|---|---|---|---|---|",
        *(
            f"| {r['class']} | {int(r['support'])} | {r['precision']:.2f} | {r['recall']:.2f} | {r['f1']:.2f} |"
            for _, r in body.iterrows()
        ),
    ]
    (reports / "classifier_summary_probe.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
