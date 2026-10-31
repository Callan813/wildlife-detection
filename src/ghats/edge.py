"""Phase 6: running the models on a camera-side device with ONNX Runtime.

Everything here needs only numpy and Pillow (plus onnxruntime to run a model), so the
same code runs on a laptop, a GitHub ARM64 runner or a Raspberry Pi, with no torch.

The detector is exported without its post-processing: the ONNX graph returns per-anchor
class scores (sigmoid applied) and boxes in 640x640 letterbox pixels, and the letterbox,
NMS and box clean-up below are a numpy copy of what PyTorch-Wildlife does.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image

from ghats.detector import Detection, _to_detections

DET_SIZE = 640
PAD_COLOUR = (114, 114, 114)


def letterbox(im: Image.Image, size: int = DET_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Same as PyTorch-Wildlife's PadAndResize + to_tensor: (CHW float32 in [0, 1], [scale, left, top]).

    The image is scaled to fit, keeping its aspect ratio, and centred on grey padding.
    """
    w, h = im.size
    scale = min(size / w, size / h)
    nw, nh = int(w * scale), int(h * scale)
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas = Image.new("RGB", (size, size), PAD_COLOUR)
    canvas.paste(im.convert("RGB").resize((nw, nh), Image.Resampling.LANCZOS), (left, top))
    x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return x, np.array([scale, left, top], dtype=np.float32)


def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    """Greedy NMS, highest score first. Returns kept indices in score order (like torchvision's nms)."""
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        order = order[1:][iou_one_to_many(boxes[i], boxes[order[1:]]) <= iou]
    return np.array(keep, dtype=np.int64)


def decode(
    scores: np.ndarray, boxes: np.ndarray, min_conf: float = 0.01, iou: float = 0.5, max_boxes: int = 300
) -> np.ndarray:
    """One image's raw outputs ([N, C] scores, [N, 4] xyxy) -> rows [cls, x1, y1, x2, y2, conf].

    Per-class NMS over every (anchor, class) above min_conf, as bbox_nms in PyTorch-Wildlife.
    """
    anchor, cls = np.nonzero(scores > min_conf)
    conf = scores[anchor, cls]
    cand = boxes[anchor]
    keep = []
    for c in np.unique(cls):
        idx = np.flatnonzero(cls == c)
        keep.append(idx[nms(cand[idx], conf[idx], iou)])
    if not keep:
        return np.zeros((0, 6), dtype=np.float32)
    keep = np.concatenate(keep)
    keep = keep[np.argsort(-conf[keep], kind="stable")][:max_boxes]
    return np.column_stack([cls[keep], cand[keep], conf[keep]]).astype(np.float32)


def to_detections(rows: np.ndarray, rev: np.ndarray, w: int, h: int) -> list[Detection]:
    """Undo the letterbox and normalise, dropping boxes that lie wholly in the padding."""
    scale, left, top = rev
    xyxy = (rows[:, 1:5] - np.array([left, top, left, top], dtype=np.float32)) / scale
    return _to_detections(rows[:, 0], rows[:, 5], xyxy, w, h)


def payload_bytes(n_crops: Sequence[int] | np.ndarray, crop_kb: float, frame_kb: float, mode: str) -> float:
    """Bytes a camera sends for a set of frames, in KB: every frame, kept frames, or crops only."""
    n = np.asarray(n_crops)
    if mode == "all_frames":
        return float(len(n) * frame_kb)
    if mode == "kept_frames":
        return float((n > 0).sum() * frame_kb)
    if mode == "crops":
        return float(n.sum() * crop_kb)
    raise ValueError(f"unknown mode {mode!r}")
