"""Phase 2, step 3: score the val/test crops with off-the-shelf classifiers.

    python scripts/08_run_baselines.py --model speciesnet
    python scripts/08_run_baselines.py --model ind-ads

Models and label mappings are in the `baselines:` section of configs/phase2.yaml; download
the weights into its `dir` first. Both models were trained on tight detector boxes, so each
box is cut again from the original frame without our 10% padding. Writes
outputs/phase2/baselines/<model>/predictions.csv with the same columns as our classifier's
predictions (p_<group> for every group, plus p_other), ready for scripts/09_event_eval.py.

The weights are torch.fx GraphModules (SpeciesNet converted from ONNX with onnx2torch, which
must be installed), which torch can only load with weights_only=False: loading them runs
pickled code, so only use files from the sources named in the config.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ghats.baselines import OTHER, TightCrops, group_probs, speciesnet_group, speciesnet_preprocess
from ghats.config import load_config


class IndPreprocess:
    """The Central India head's pipeline: resize, ImageNet-style normalise, NCHW (the model permutes)."""

    def __init__(self, size: int, mean, std):
        from torchvision import transforms as T

        self.tf = T.Compose([T.Resize((size, size), antialias=True), T.ToTensor(), T.Normalize(mean, std)])

    def __call__(self, crop: Image.Image):
        return self.tf(crop)


def load_speciesnet(b: dict, device: str):
    import torch

    d = Path(b["dir"])
    model = torch.load(d / b["weights"], map_location=device, weights_only=False).eval()
    labels = [line.strip() for line in (d / b["labels"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    groups = [speciesnet_group(label, b["rules"]) for label in labels]
    return model, groups, speciesnet_preprocess


def load_ind(b: dict, device: str):
    import torch
    import torch.nn.functional as F

    d = Path(b["dir"])
    ckpt = torch.load(d / b["weights"], map_location="cpu", weights_only=True)
    backbone = torch.load(d / b["backbone"], map_location="cpu", weights_only=False)

    class Head(torch.nn.Module):
        """Frozen SpeciesNet backbone + linear head, as in Addax's inference.py."""

        def __init__(self):
            super().__init__()
            self.backbone = backbone
            with torch.no_grad():
                z = backbone(torch.zeros(1, ckpt["img_size"], ckpt["img_size"], 3))
            self.head = torch.nn.Linear(self.pool(z).shape[1], ckpt["num_classes"])

        @staticmethod
        def pool(z):
            return F.adaptive_avg_pool2d(z, 1).flatten(1) if z.ndim == 4 else z.flatten(1)

        def forward(self, x):
            if ckpt["input_layout"].lower() == "nhwc":
                x = x.permute(0, 2, 3, 1).contiguous()
            return self.head(self.pool(self.backbone(x)))

    model = Head()
    model.load_state_dict(ckpt["model"])
    groups = [b["classes"].get(name, OTHER) for name in ckpt["class_names"]]
    norm = ckpt["normalize"]
    return model.to(device).eval(), groups, IndPreprocess(ckpt["img_size"], norm["mean"], norm["std"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["speciesnet", "ind-ads"])
    ap.add_argument("--config", default="configs/phase2.yaml")
    ap.add_argument("--crops", default="data/crops/crops.csv")
    ap.add_argument("--manifest", default="data/swg/manifest_phase2.csv")
    ap.add_argument("--ours", default="outputs/phase2/classifier/predictions.csv", help="For the group list")
    ap.add_argument("--out", default="outputs/phase2/baselines")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="Only N crops (quick smoke test)")
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader

    b = load_config(args.config)["baselines"][args.model]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    groups = sorted(pd.read_csv(args.ours, nrows=0).filter(regex="^p_").columns.str[2:])

    crops = pd.read_csv(args.crops, dtype={"image_id": str})
    crops = crops[crops["split"].isin(["val", "test"])]
    paths = pd.read_csv(args.manifest, dtype={"image_id": str}).set_index("image_id")["local_path"]
    crops = crops.assign(local_path=crops["image_id"].map(paths)).dropna(subset=["local_path"])
    if args.limit:
        crops = crops.head(args.limit)

    model, class_groups, preprocess = (load_speciesnet if args.model == "speciesnet" else load_ind)(b, device)
    print(f"{args.model}: {len(class_groups)} classes -> {len(set(class_groups) - {OTHER})} of our groups")
    print(f"{len(crops):,} val/test crops on {device}")

    loader = DataLoader(
        TightCrops(crops["local_path"].tolist(), [ast.literal_eval(s) for s in crops["bbox"]], preprocess),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device == "cuda",
    )
    probs = []
    with torch.no_grad():
        for i, x in enumerate(loader):
            probs.append(torch.softmax(model(x.to(device, non_blocking=True)).float(), 1).cpu().numpy())
            if i % 50 == 0:
                print(f"  {min((i + 1) * args.batch_size, len(crops)):,}/{len(crops):,}", flush=True)
    grouped = group_probs(np.concatenate(probs), class_groups, groups)

    names = [*groups, OTHER]
    out = crops[["crop_id", "image_id", "seq_key", "location", "split", "label", "train_ok", "conf"]].reset_index(
        drop=True
    )
    out["pred"] = [names[k] for k in grouped.argmax(1)]
    out["pred_prob"] = grouped.max(1)
    out = pd.concat([out, pd.DataFrame(grouped, columns=[f"p_{g}" for g in names])], axis=1)
    dest = Path(args.out) / args.model
    dest.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest / "predictions.csv", index=False)
    print(f"Wrote {dest / 'predictions.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
