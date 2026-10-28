"""Leopard re-ID model: DINOv2 backbone + embedding neck, ArcFace loss, crop loading.

Needs torch; used by scripts/14_train_reid.py (training) and 15_open_set.py (scoring).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ghats.reid import l2_normalise


class Crops(Dataset):
    def __init__(self, paths: list[Path], labels: list[int], tf):
        self.paths, self.labels, self.tf = paths, labels, tf

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert("RGB")), self.labels[i]


def transforms(size: tuple[int, int], mean, std, grayscale_p: float, train: bool):
    from torchvision import transforms as T

    if not train:
        return T.Compose([T.Resize(size), T.ToTensor(), T.Normalize(mean, std)])
    return T.Compose(
        [
            T.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(size[1] / size[0] * 0.8, size[1] / size[0] * 1.25)),
            T.ColorJitter(0.3, 0.3, 0.2),
            T.RandomGrayscale(grayscale_p),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )


class ArcFace(nn.Module):
    """Additive angular margin: logit of the true class is cos(theta + m), all scaled by s."""

    def __init__(self, dim: int, n_classes: int, margin: float, scale: float):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_classes, dim) * 0.01)
        self.m, self.s = margin, scale

    def forward(self, emb: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cos = F.linear(F.normalize(emb), F.normalize(self.w)).clamp(-1 + 1e-7, 1 - 1e-7)
        target = torch.cos(torch.acos(cos.gather(1, y[:, None])) + self.m)
        logits = cos.scatter(1, y[:, None], target)
        return F.cross_entropy(self.s * logits, y)


class Embedder(nn.Module):
    def __init__(self, backbone: str, dim: int):
        super().__init__()
        import timm

        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0, dynamic_img_size=True)
        self.neck = nn.Sequential(nn.Linear(self.backbone.num_features, dim), nn.BatchNorm1d(dim))

    def forward(self, x):
        return self.neck(self.backbone(x))


@torch.no_grad()
def embed(model: Embedder, loader: DataLoader, device: str, raw: bool = False) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in loader:
        with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            f = model.backbone(x.to(device)) if raw else model(x.to(device))
        out.append(f.float().cpu().numpy())
    return l2_normalise(np.concatenate(out))


def load_trained(path: Path, device: str) -> Embedder:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = Embedder(ckpt["config"]["backbone"], ckpt["config"]["embed_dim"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval()


def eval_loader(paths: list[Path], cfg: dict, model: Embedder, batch_size: int = 32, workers: int = 4) -> DataLoader:
    """Loader with the training config's eval transform (resize to cfg size, no augmentation)."""
    pc = model.backbone.pretrained_cfg
    tf = transforms(tuple(cfg["size"]), pc["mean"], pc["std"], 0, train=False)
    return DataLoader(Crops(paths, [0] * len(paths), tf), batch_size=batch_size, num_workers=workers)
