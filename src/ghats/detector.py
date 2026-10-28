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
        # PyTorch-Wildlife 1.3's YOLO-MIT single_image_detection() reloads the
        # checkpoint on every call (~1 s/image, GPU mostly idle). For that model we
        # load once here and run batches ourselves; other models use detect().
        self.batched = cls_name == "MegaDetectorV6MIT"
        if self.batched:
            m = self.model
            m.cfg.task.nms.min_confidence = conf_floor
            m._load_model(weights=m.weights, device=self.device, url=m.url)
            m.model.eval()

    @property
    def transform(self) -> Any:
        """Letterbox transform for load_image(); picklable, so DataLoader workers can use it."""
        return self.model.transform

    def preprocess(self, image_path: str) -> tuple[Any, Any, tuple[int, int]]:
        return load_image(image_path, self.transform)

    def detect_batch(self, images: Any, revs: Any, sizes: list[tuple[int, int]]) -> list[list[Detection]]:
        """Run on a batch from preprocess(): images [B,3,640,640], revs [B,5], sizes [(w, h)]."""
        import torch

        with torch.no_grad():
            pred = self.model.model(images.to(self.device, non_blocking=True))
            per_image = self.model.post_proccess(pred, revs.to(self.device))
        out = []
        for p, (w, h) in zip(per_image, sizes):
            p = p.cpu().numpy()  # rows: [cls, x1, y1, x2, y2, conf] in original-image pixels
            out.append(_to_detections(p[:, 0], p[:, 5], p[:, 1:5], w, h))
        return out

    def detect(self, image_path: str) -> tuple[list[Detection], float]:
        """Run on one image. Returns (detections, latency in ms)."""
        start = time.perf_counter()
        if self.batched:
            image, rev, size = self.preprocess(image_path)
            dets = self.detect_batch(image[None], rev[None], [size])[0]
            return dets, (time.perf_counter() - start) * 1000.0
        res: dict[str, Any] = self.model.single_image_detection(image_path, det_conf_thres=self.conf_floor)
        ms = (time.perf_counter() - start) * 1000.0
        dets = res["detections"]
        with Image.open(image_path) as im:
            w, h = im.size
        return _to_detections(dets.class_id, dets.confidence, dets.xyxy, w, h), ms


def load_image(image_path: str, transform: Any) -> tuple[Any, Any, tuple[int, int]]:
    """Load and letterbox one image: (tensor [3,640,640], reverse transform [5], (w, h))."""
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        image, _, rev = transform(im)
        return image, rev, im.size


def has_area(bbox: list[float]) -> bool:
    return bbox[2] > bbox[0] and bbox[3] > bbox[1]


def _to_detections(classes: Any, confs: Any, boxes: Any, w: int, h: int) -> list[Detection]:
    # PyTorch-Wildlife 1.3 returns pixel boxes (xyxy) in original-image
    # coordinates, sometimes slightly past the edge; normalise and clamp.
    # The model also fires on the letterbox padding above and below wide frames
    # (conf up to ~0.2, in almost every image). Those boxes lie wholly outside the
    # image, clamp to zero-height strips on the border, and are dropped.
    dets = [
        Detection(
            category=_CLASS_NAMES.get(int(c), str(int(c))),
            conf=round(float(p), 4),
            bbox=[round(min(max(float(v) / s, 0.0), 1.0), 4) for v, s in zip(box, (w, h, w, h))],
        )
        for c, p, box in zip(classes, confs, boxes)
    ]
    return [d for d in dets if has_area(d.bbox)]
