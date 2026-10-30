"""Phase 3, step 3: fine-tune DINOv2-S for leopard re-ID with an ArcFace loss.

    python scripts/14_train_reid.py --config configs/phase3.yaml

Off-the-shelf features (scripts/13_reid_baseline.py) know "leopard", not "which leopard".
ArcFace trains the embedding so crops of one leopard's flank sit close together and
other leopards' flanks are pushed a margin away. Training uses only the train leopards;
after every epoch the val leopards are scored with the same retrieval rules as the
baseline (ghats.reid), and the best epoch by val top-1 per leopard is kept. Epoch 0 is the
untrained backbone at the same input size, for a fair before/after.

Writes outputs/phase3/reid/{best.pt,history.csv} and reports/phase3/reid_finetune.md.

--leopard102 adds every LeopardID102 leopard (crops from 17_leopard102_crops.py, flanks
from 18_flank_classifier.py) to the training set as extra classes. Val and test stay the
Leopard ID 2022 leopards, so the test numbers are directly comparable. Use --tag and a
separate --out so the original model and report are kept, e.g.

    python scripts/14_train_reid.py --leopard102 --out outputs/phase3/reid_lp102 --tag _lp102
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ghats.config import load_config
from ghats.reid import retrieval_metrics
from ghats.reid_model import ArcFace, Crops, Embedder, embed, transforms


def score(emb: np.ndarray, crops: pd.DataFrame, ks) -> dict:
    return retrieval_metrics(emb, crops["name"], crops["encounter"], crops["viewpoint"], ks)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase3.yaml")
    ap.add_argument("--out", default="outputs/phase3/reid")
    ap.add_argument("--reports", default="reports/phase3")
    ap.add_argument("--epochs", type=int, help="Override the config")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--leopard102", action="store_true", help="Add LeopardID102 leopards to training")
    ap.add_argument("--tag", default="", help="Suffix for the report file names")
    args = ap.parse_args()

    cfg = load_config(args.config)
    t, ks = cfg["train"], cfg["reid"]["ks"]
    epochs = t["epochs"] if args.epochs is None else args.epochs
    size = tuple(t["size"])
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    crop_dir = Path(cfg["crops"]["out"])
    crops = pd.read_csv(crop_dir / "crops.csv")
    crops["path"] = [crop_dir / "images" / f"{c}.jpg" for c in crops["crop_id"]]
    train = crops[crops["split"] == "train"].reset_index(drop=True)
    val = crops[crops["split"] == "val"].reset_index(drop=True)
    test = crops[crops["split"] == "test"].reset_index(drop=True)
    if args.leopard102:
        lp_dir = Path(cfg["leopard102"]["out"])
        lp = pd.read_csv(lp_dir / "crops.csv", keep_default_na=False)
        lp["path"] = [lp_dir / "images" / f"{c}.jpg" for c in lp["crop_id"]]
        lp["name"] = "lp102_" + lp["name"]
        train = pd.concat([train, lp[["name", "viewpoint", "path"]]], ignore_index=True)
    flank = train["name"] + "/" + train["viewpoint"]
    classes = {c: k for k, c in enumerate(sorted(flank.unique()))}
    print(
        f"train: {len(train):,} crops, {train['name'].nunique()} leopards, {len(classes)} flanks; val {len(val):,}; test {len(test):,}"
    )

    model = Embedder(t["backbone"], t["embed_dim"]).to(device)
    data_cfg = model.backbone.pretrained_cfg
    mean, std = data_cfg["mean"], data_cfg["std"]
    tf_eval = transforms(size, mean, std, 0, train=False)

    def loader(df: pd.DataFrame, labels, tf, shuffle: bool) -> DataLoader:
        return DataLoader(
            Crops(df["path"].tolist(), labels, tf),
            batch_size=t["batch_size"],
            shuffle=shuffle,
            drop_last=shuffle,
            num_workers=args.workers,
            pin_memory=device == "cuda",
            persistent_workers=args.workers > 0,
        )

    train_dl = loader(train, flank.map(classes).tolist(), transforms(size, mean, std, t["grayscale_p"], True), True)
    val_dl = loader(val, [0] * len(val), tf_eval, False)
    test_dl = loader(test, [0] * len(test), tf_eval, False)

    arc = ArcFace(t["embed_dim"], len(classes), t["arcface_margin"], t["arcface_scale"]).to(device)
    opt = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": t["lr_backbone"]},
            {"params": [*model.neck.parameters(), *arc.parameters()], "lr": t["lr_head"]},
        ],
        weight_decay=t["weight_decay"],
    )
    steps = epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    before = score(embed(model, val_dl, device, raw=True), val, ks)
    history = [{"epoch": 0, "loss": float("nan"), **{f"val_{k}": v for k, v in before.items()}}]
    print(f"epoch  0  (untrained backbone)  val top1/leopard {before['top1_per_id']:.3f}  mAP {before['mAP']:.3f}")
    best = before["top1_per_id"]
    for epoch in range(1, epochs + 1):
        model.train()
        start, losses = time.time(), []
        for x, y in train_dl:
            with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                emb = model(x.to(device, non_blocking=True))
            loss = arc(emb.float(), y.to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        m = score(embed(model, val_dl, device), val, ks)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in m.items()}})
        flag = ""
        if m["top1_per_id"] > best:
            best, flag = m["top1_per_id"], "  *"
            torch.save({"model": model.state_dict(), "config": t}, out / "best.pt")
        print(
            f"epoch {epoch:2d}  loss {np.mean(losses):.3f}  val top1/leopard {m['top1_per_id']:.3f}  "
            f"mAP {m['mAP']:.3f}  ({time.time() - start:.0f} s){flag}",
            flush=True,
        )
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)

    # Test once: the untrained backbone and the best epoch, same input size.
    untrained = Embedder(t["backbone"], t["embed_dim"]).to(device)
    test_before = score(embed(untrained, test_dl, device, raw=True), test, ks)
    if (out / "best.pt").exists():
        model.load_state_dict(torch.load(out / "best.pt", map_location=device)["model"])
        test_after = score(embed(model, test_dl, device), test, ks)
        best_epoch = int(pd.DataFrame(history).iloc[1:]["val_top1_per_id"].idxmax())
    else:
        test_after, best_epoch = test_before, 0

    cols = [f"top{k}_per_id" for k in ks] + ["mAP", "top1"]
    heads = [f"Top-{k}" for k in ks] + ["mAP", "Top-1 per query"]
    lines = [
        "# Phase 3: DINOv2-S fine-tuned with ArcFace" + (" + LeopardID102 leopards" if args.leopard102 else ""),
        "",
        (
            f"Trained on {len(train):,} crops of {train['name'].nunique()} leopards ({len(classes)} flanks), "
            f"{epochs} epochs, input {size[0]}x{size[1]}, seed {args.seed}; best epoch {best_epoch} by val top-1 per leopard. "
            f"Test: {len(test):,} crops of {test['name'].nunique()} leopards never seen in training, scored as in "
            "reid_baseline.md (no same-encounter matches, same flank only, top-k averaged per leopard)."
        ),
        "",
        "| Model | " + " | ".join(heads) + " |",
        "|---|" + "---|" * len(cols),
        "| DINOv2-S, untrained (same input size) | " + " | ".join(f"{test_before[c]:.1%}" for c in cols) + " |",
        "| DINOv2-S + ArcFace | " + " | ".join(f"{test_after[c]:.1%}" for c in cols) + " |",
        "",
        f"Per-epoch val scores are in {Path(args.out).as_posix()}/history.csv.",
    ]
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"reid_finetune{args.tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame(history).to_csv(reports / f"reid_finetune{args.tag}_history.csv", index=False)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
