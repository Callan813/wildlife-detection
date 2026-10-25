"""Thin wrapper around PyTorch-Wildlife's MegaDetector V6 models.

Imports torch lazily so the rest of the package (metrics, events, tests) works
without a GPU stack installed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from PIL import Image

ANIMAL, PERSON, VEHICLE = 0, 1, 2
_CLASS_NAMES = {ANIMAL: "animal", PERSON: "person", VEHICLE: "vehicle"}
_SUPPORTED = ("MegaDetectorV6MIT", "MegaDetectorV6", "MegaDetectorV6Apache")


@dataclass
class Detection:
    category: str
    conf: float
    bbox: list[float]  # normalised [x1, y1, x2, y2]


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class MegaDetector:
    def __init__(self, cls_name: str, version: str, device: str = "auto", conf_floor: float = 0.01):
        if cls_name not in _SUPPORTED:
            raise ValueError(f"detector.class must be one of {_SUPPORTED}, got {cls_name!r}")
        from PytorchWildlife.models import detection as pw_detection

        self.device = resolve_device(device)
        self.conf_floor = conf_floor
        self.model = getattr(pw_detection, cls_name)(device=self.device, pretrained=True, version=version)
        self.name = f"{cls_name}/{version}"

    def detect(self, image_path: str) -> tuple[list[Detection], float]:
        """Run on one image. Returns (detections, latency in ms)."""
        start = time.perf_counter()
        res: dict[str, Any] = self.model.single_image_detection(image_path, det_conf_thres=self.conf_floor)
        ms = (time.perf_counter() - start) * 1000.0
        dets = res["detections"]
        # PyTorch-Wildlife 1.3 returns pixel boxes (xyxy) in original-image
        # coordinates, sometimes slightly past the edge; normalise and clamp.
        with Image.open(image_path) as im:
            w, h = im.size
        out = [
            Detection(
                category=_CLASS_NAMES.get(int(c), str(int(c))),
                conf=round(float(p), 4),
                bbox=[round(min(max(float(v) / s, 0.0), 1.0), 4) for v, s in zip(box, (w, h, w, h))],
            )
            for c, p, box in zip(dets.class_id, dets.confidence, dets.xyxy)
        ]
        return out, ms
