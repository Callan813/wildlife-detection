"""Phase 4, step 1: turn every frame and detector crop into a CLIP-style vector.

    python scripts/29_embed.py                  # both models, val + test cameras
    python scripts/29_embed.py --models bioclip-2

For each model this writes outputs/phase4/embeddings/<model>_frames.npz and
<model>_crops.npz: ids plus L2-normalised float16 vectors, so a text query is one matrix
product away from a score per photo. Frames also get a colour score (mean spread between
the RGB channels of a small thumbnail): infrared night shots are grey, so it is close to 0.

Full frames are letterboxed to the model's square input instead of centre-cropped, so an
animal at the edge of the frame isn't cut off. A 3264 x 2448 frame shrunk to 224 px loses
small animals, which is why the crops are embedded as well.

Weights come from outputs/phase4/models/<name> (open_clip "local-dir", safetensors only):

    huggingface_hub.snapshot_download("imageomics/bioclip-2", local_dir=".../bioclip-2",
        allow_patterns=["*.json", "*.txt", "open_clip_model.safetensors"])
    same for timm/vit_base_patch16_clip_224.openai -> .../clip-b16

Runs on the GPU when there is one (RTX 3050 laptop: about 5 minutes for both models).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ghats.config import load_config

THUMB = 64


def colour_score(im: Image.Image) -> float:
    """Mean absolute difference between the RGB channels, 0-255. IR night shots are ~0."""
    a = np.asarray(im.resize((THUMB, THUMB)), dtype=np.float32)
    return float(np.abs(a - a.mean(axis=2, keepdims=True)).mean())


class Images(Dataset):
    def __init__(self, paths: list[str], transform, size: int, colour: bool):
        self.paths, self.transform, self.size, self.colour = paths, transform, size, colour

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        im = Image.open(self.paths[i])
        im.draft("RGB", (self.size * 2, self.size * 2))  # JPEG DCT scaling: decodes 8x faster
        im = im.convert("RGB")
        c = colour_score(im) if self.colour else 0.0
        return self.transform(im), c


def load_model(mdir: Path, name: str, device: str, resize_mode: str):
    import open_clip

    model, _, pre = open_clip.create_model_and_transforms(
        f"local-dir:{mdir / name}", image_resize_mode=resize_mode, device=device
    )
    if device == "cuda":
        model = model.half()
    return model.eval(), pre


@torch.no_grad()
def embed(model, pre, paths, device, batch_size, workers, colour) -> tuple[np.ndarray, np.ndarray]:
    size = model.visual.image_size
    size = size[0] if isinstance(size, (tuple, list)) else size
    dl = DataLoader(Images(paths, pre, size, colour), batch_size=batch_size, num_workers=workers)
    vecs, cols = [], []
    dtype = next(model.parameters()).dtype
    for x, c in dl:
        f = model.encode_image(x.to(device, dtype))
        f = f / f.norm(dim=-1, keepdim=True)
        vecs.append(f.float().cpu().numpy().astype(np.float16))
        cols.append(c.numpy())
    return np.concatenate(vecs), np.concatenate(cols).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase4.yaml")
    ap.add_argument("--models", nargs="*", help="default: all in the config")
    ap.add_argument("--force", action="store_true", help="redo files that already exist")
    args = ap.parse_args()
    cfg = load_config(args.config)
    d, mc = cfg["data"], cfg["models"]

    split = pd.read_csv(d["splits"]).set_index("location")["split"]
    keep = set(d["index_splits"])
    frames = pd.read_csv(d["manifest"], dtype={"image_id": str})
    frames = frames[frames["location"].map(split).isin(keep)]
    crops = pd.read_csv(d["crops"], dtype={"image_id": str})
    crops = crops[crops["split"].isin(keep)]
    crop_paths = [str(Path(d["crop_dir"]) / f"{c}.jpg") for c in crops["crop_id"]]
    print(f"{len(frames)} frames, {len(crops)} crops from the {', '.join(sorted(keep))} cameras")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(cfg["embeddings"])
    out.mkdir(parents=True, exist_ok=True)
    for name in args.models or mc["names"]:
        jobs = [
            ("frames", frames["image_id"].tolist(), frames["local_path"].tolist(), "longest", True),
            ("crops", crops["crop_id"].tolist(), crop_paths, "longest", False),
        ]
        model = pre = None
        for kind, ids, paths, mode, colour in jobs:
            f = out / f"{name}_{kind}.npz"
            if f.exists() and not args.force:
                print(f"{f} exists, skipping")
                continue
            if model is None:
                model, pre = load_model(Path(mc["dir"]), name, device, mode)
            t = time.time()
            vecs, cols = embed(model, pre, paths, device, mc["batch_size"], mc["workers"], colour)
            np.savez(f, ids=np.array(ids), vecs=vecs, colour=cols)
            print(f"{name} {kind}: {len(ids)} in {time.time() - t:.0f} s -> {f} {vecs.shape}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
