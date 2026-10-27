"""Phase 2, step 2: fine-tune a pretrained image model on the detector crops.

    python scripts/07_train_classifier.py --config configs/phase2.yaml --crops data/crops \
        --out outputs/phase2/classifier

Trains on train-split crops with train_ok=True, keeps the epoch with the best
validation macro F1, then scores every val and test crop (including the
low-confidence ones kept for evaluation). Writes:
  <out>/best.pt             weights, class names, model name
  <out>/history.csv         loss and macro F1 per epoch
  <out>/predictions.csv     one row per val/test crop: true label, predicted label, class probabilities
  reports/phase2/           per-class test metrics, confusion matrix, summary
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ghats.classify import class_weights, confusion_matrix, per_class_report
from ghats.config import load_config


class CropDataset:
    """Map-style dataset for torch's DataLoader: (image tensor, class index)."""

    def __init__(self, paths: list[str], targets: list[int], transform):
        self.paths = paths
        self.targets = targets
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            return self.transform(im.convert("RGB")), self.targets[i]


def transforms(size: int, mean, std):
    from torchvision import transforms as T

    train = T.Compose(
        [
            # Random zoom and aspect change: the detector's boxes are never framed the same way twice.
            T.RandomResizedCrop(size, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            # About half of camera-trap frames are infrared night shots, which are grey.
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    test = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(mean, std)])
    return train, test


def lr_at(step: int, total: int, warmup: int) -> float:
    """Multiplier on the base learning rate: linear warmup, then cosine decay to 0."""
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def predict(model, loader, device: str):
    import torch

    model.eval()
    probs = []
    with torch.no_grad(), torch.autocast(device_type=device, enabled=device == "cuda"):
        for x, _ in loader:
            probs.append(model(x.to(device, non_blocking=True)).float().softmax(1).cpu())
    return torch.cat(probs).numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase2.yaml")
    ap.add_argument("--crops", default="data/crops")
    ap.add_argument("--out", default="outputs/phase2/classifier")
    ap.add_argument("--reports", default="reports/phase2")
    ap.add_argument("--epochs", type=int, help="Override the config")
    ap.add_argument("--limit", type=int, help="Use only N crops per split (quick smoke test)")
    ap.add_argument("--workers", type=int, default=4, help="Image decoding processes")
    args = ap.parse_args()

    import timm
    import torch
    from torch.utils.data import DataLoader

    c = load_config(args.config)["classifier"]
    epochs = args.epochs or c["epochs"]
    random.seed(c["seed"])
    np.random.seed(c["seed"])
    torch.manual_seed(c["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    crops_dir = Path(args.crops)
    crops = pd.read_csv(crops_dir / "crops.csv")
    crops["path"] = [str(crops_dir / "images" / f"{cid}.jpg") for cid in crops["crop_id"]]
    classes = sorted(crops["label"].unique())
    idx = {name: k for k, name in enumerate(classes)}
    crops["target"] = crops["label"].map(idx)

    train = crops[(crops["split"] == "train") & crops["train_ok"]]
    val_ok = crops[(crops["split"] == "val") & crops["train_ok"]]
    if args.limit:
        train = train.sample(min(args.limit, len(train)), random_state=c["seed"])
        val_ok = val_ok.sample(min(args.limit, len(val_ok)), random_state=c["seed"])
    counts = train["target"].value_counts().reindex(range(len(classes)), fill_value=0)
    print(f"{len(classes)} classes, {len(train):,} train crops, {len(val_ok):,} val crops, device {device}")

    model = timm.create_model(c["model"], pretrained=True, num_classes=len(classes)).to(device)
    cfg = timm.data.resolve_data_config({}, model=model)
    tf_train, tf_test = transforms(c["image_size"], cfg["mean"], cfg["std"])

    def loader(df: pd.DataFrame, tf, shuffle: bool) -> DataLoader:
        return DataLoader(
            CropDataset(df["path"].tolist(), df["target"].tolist(), tf),
            batch_size=c["batch_size"],
            shuffle=shuffle,
            num_workers=args.workers,
            pin_memory=device == "cuda",
            persistent_workers=args.workers > 0,
            drop_last=shuffle,
        )

    train_dl = loader(train, tf_train, shuffle=True)
    val_dl = loader(val_ok, tf_test, shuffle=False)

    weights = torch.tensor(class_weights(counts.clip(lower=1), c["class_weight_power"]), dtype=torch.float32)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=c["label_smoothing"])
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    total = epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_at(s, total, int(c["warmup_epochs"] * len(train_dl))))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    history, best_f1 = [], -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        start, run_loss, n = time.perf_counter(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * len(y)
            n += len(y)

        probs = predict(model, val_dl, device)
        rep = per_class_report(val_ok["target"], probs.argmax(1), classes)
        f1 = float(rep.iloc[-1]["f1"])
        secs = time.perf_counter() - start
        history.append({"epoch": epoch, "train_loss": run_loss / n, "val_macro_f1": f1, "seconds": round(secs)})
        print(f"epoch {epoch:2d}/{epochs}  loss {run_loss / n:.3f}  val macro F1 {f1:.3f}  ({secs:.0f} s)")
        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model": c["model"], "classes": classes, "state_dict": model.state_dict()}, out / "best.pt")
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)

    # Score every val and test crop with the best epoch, including the low-confidence ones.
    model.load_state_dict(torch.load(out / "best.pt", weights_only=False)["state_dict"])
    held_out = crops[crops["split"].isin(["val", "test"])]
    if args.limit:
        held_out = held_out.groupby("split").head(args.limit)
    probs = predict(model, loader(held_out, tf_test, shuffle=False), device)
    pred = held_out[["crop_id", "image_id", "seq_key", "location", "split", "label", "train_ok", "conf"]].copy()
    pred["pred"] = [classes[k] for k in probs.argmax(1)]
    pred["pred_prob"] = probs.max(1)
    pred = pd.concat([pred.reset_index(drop=True), pd.DataFrame(probs, columns=[f"p_{n}" for n in classes])], axis=1)
    pred.to_csv(out / "predictions.csv", index=False)

    test = pred[(pred["split"] == "test") & pred["train_ok"]]
    y_true, y_pred = test["label"].map(idx), test["pred"].map(idx)
    rep = per_class_report(y_true, y_pred, classes)
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, len(classes)), index=classes, columns=classes)

    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    rep.to_csv(reports / "classifier_test.csv", index=False)
    cm.to_csv(reports / "confusion_test.csv")
    worst = rep.iloc[:-1][rep.iloc[:-1]["support"] > 0].sort_values("f1")
    n_cams = test["location"].nunique()
    acc = (y_true == y_pred).mean()
    lines = [
        "# Phase 2 results: species classifier on detector crops",
        "",
        f"Model: `{c['model']}`, fine-tuned {epochs} epochs on {len(train):,} crops from train cameras.",
        f"Test: {len(test):,} crops (detector conf >= 0.2 or background) from {n_cams} cameras never seen in training.",
        "",
        f"**Test macro F1 {rep.iloc[-1]['f1']:.3f}**, accuracy {acc:.3f} (best val macro F1 {best_f1:.3f}).",
        "",
        "| Class | Test crops | Precision | Recall | F1 |",
        "|---|---|---|---|---|",
        *(
            f"| {r['class']} | {r['support']} | {r['precision']:.2f} | {r['recall']:.2f} | {r['f1']:.2f} |"
            for _, r in worst.iterrows()
        ),
        "",
        "Classes with fewer than ~50 test crops have noisy F1: one or two crops move it by several points.",
    ]
    (reports / "classifier_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
