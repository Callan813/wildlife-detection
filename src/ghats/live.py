"""Phase 6: the camera-side pipeline, frame by frame.

    frame -> detector (ONNX) -> animal boxes -> species model (ONNX) + probe head (numpy)
          -> per-frame result -> trigger events -> keep / drop, and what to send

Like ghats.edge, this needs only numpy, Pillow and onnxruntime, so it runs unchanged on a
Raspberry Pi. Reading video or a webcam (scripts/26_live.py) adds OpenCV; still images don't.

The operating point is Phase 2's (reports/phase2/event_eval.csv, "probe, best box" at the
90% target, picked on val cameras): boxes from 0.02, and an event is kept when its best box
has P(not background) >= 0.273. On unseen test cameras that kept 91% of animal events and
dropped 74% of empty ones. A camera at a new site should recalibrate it (Phase 3 and 5
showed thresholds don't transfer well).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from ghats.crops import BACKGROUND
from ghats.detector import Detection
from ghats.edge import decode, letterbox, to_detections

SPECIES_SIZE = 480


def session(path: Path | str, threads: int):
    import onnxruntime as ort

    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])


def resize_bilinear(x: np.ndarray, size: int) -> np.ndarray:
    """HWC float array -> size x size, bilinear without antialiasing (torch's
    interpolate(mode="bilinear", align_corners=False), which SpeciesNet's preprocessing uses)."""

    def axis(n_in: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        src = np.maximum((np.arange(size) + 0.5) * (n_in / size) - 0.5, 0.0)
        lo = np.minimum(np.floor(src).astype(np.int64), n_in - 1)
        hi = np.minimum(lo + 1, n_in - 1)
        return lo, hi, (src - lo).astype(np.float32)

    y0, y1, wy = axis(x.shape[0])
    x0, x1, wx = axis(x.shape[1])
    rows = x[y0] * (1 - wy)[:, None, None] + x[y1] * wy[:, None, None]
    return rows[:, x0] * (1 - wx)[None, :, None] + rows[:, x1] * wx[None, :, None]


def species_input(crop: Image.Image) -> np.ndarray:
    """numpy copy of ghats.baselines.speciesnet_preprocess: 480x480 HWC in [0, 1], with the
    uint8 round trip torchvision's convert_image_dtype does."""
    x = np.asarray(crop.convert("RGB"), dtype=np.float32) / 255.0
    x = resize_bilinear(x, SPECIES_SIZE)
    return np.floor(x * 255.999).astype(np.uint8).astype(np.float32) / 255.0


def cut(im: Image.Image, bbox: list[float]) -> Image.Image:
    """The tight detector box, as the species features were computed in Phase 2."""
    x1, y1, x2, y2 = bbox
    box = (int(x1 * im.width), int(y1 * im.height), int(x2 * im.width), int(y2 * im.height))
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        box = (0, 0, im.width, im.height)
    return im.crop(box)


class SpeciesHead:
    """The Phase 2 probe (StandardScaler + LogisticRegression) as plain arrays."""

    def __init__(self, path: Path | str):
        z = np.load(path)
        self.mean, self.scale = z["mean"], z["scale"]
        self.coef, self.intercept = z["coef"], z["intercept"]
        self.classes = [str(c) for c in z["classes"]]

    def probs(self, features: np.ndarray) -> np.ndarray:
        logits = ((features - self.mean) / self.scale) @ self.coef.T + self.intercept
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)


@dataclass
class Box:
    bbox: list[float]  # normalised x1, y1, x2, y2
    det_conf: float
    species: str  # best non-background group
    species_prob: float
    p_animal: float  # 1 - P(background)


@dataclass
class FrameResult:
    t: float  # seconds (from EXIF, file time or video position)
    boxes: list[Box]
    ms: dict[str, float]

    @property
    def p_animal(self) -> float:
        return max((b.p_animal for b in self.boxes), default=0.0)


class Pipeline:
    def __init__(
        self,
        detector: Path | str,
        species: Path | str,
        head: Path | str,
        threads: int = 4,
        min_conf: float = 0.02,
        max_boxes: int = 5,
    ):
        self.det = session(detector, threads)
        self.sp = session(species, threads)
        self.head = SpeciesHead(head)
        self.min_conf, self.max_boxes = min_conf, max_boxes
        self.bg = self.head.classes.index(BACKGROUND)

    def detect(self, im: Image.Image) -> list[Detection]:
        x, rev = letterbox(im)
        scores, boxes = self.det.run(None, {"input": x[None]})
        dets = to_detections(decode(scores[0], boxes[0]), rev, im.width, im.height)
        animal = [d for d in dets if d.category == "animal" and d.conf >= self.min_conf]
        return sorted(animal, key=lambda d: -d.conf)[: self.max_boxes]

    def classify(self, im: Image.Image, dets: list[Detection]) -> np.ndarray:
        if not dets:
            return np.zeros((0, len(self.head.classes)), dtype=np.float32)
        x = np.stack([species_input(cut(im, d.bbox)) for d in dets])
        return self.head.probs(self.sp.run(None, {"input": x})[0])

    def __call__(self, im: Image.Image, t: float) -> FrameResult:
        import time

        im = im.convert("RGB")
        t0 = time.perf_counter()
        dets = self.detect(im)
        t1 = time.perf_counter()
        probs = self.classify(im, dets)
        t2 = time.perf_counter()
        boxes = []
        for d, p in zip(dets, probs):
            animal = np.delete(p, self.bg)
            k = int(animal.argmax())
            names = [c for i, c in enumerate(self.head.classes) if i != self.bg]
            boxes.append(
                Box(d.bbox, d.conf, names[k], float(animal[k] / max(animal.sum(), 1e-9)), 1 - float(p[self.bg]))
            )
        return FrameResult(t, boxes, {"detector": (t1 - t0) * 1000, "species": (t2 - t1) * 1000})


@dataclass
class Event:
    """One camera trigger: frames no more than ``gap`` seconds apart."""

    frames: list[FrameResult] = field(default_factory=list)
    names: list[str] = field(default_factory=list)  # frame name or video timestamp, for the log

    @property
    def start(self) -> float:
        return self.frames[0].t

    @property
    def end(self) -> float:
        return self.frames[-1].t

    def best(self) -> tuple[int, Box] | None:
        """The frame index and box the event is judged by: the highest P(not background)."""
        cands = [(i, b) for i, f in enumerate(self.frames) for b in f.boxes]
        return max(cands, key=lambda c: c[1].p_animal, default=None)

    def species(self) -> tuple[str, float]:
        """Group with the most probability summed over the event's animal boxes (each weighted by
        its P(not background)), so one blurry frame doesn't decide it; share of that sum."""
        votes: dict[str, float] = {}
        for f in self.frames:
            for b in f.boxes:
                votes[b.species] = votes.get(b.species, 0.0) + b.p_animal * b.species_prob
        if not votes:
            return "", 0.0
        top = max(votes, key=votes.__getitem__)
        return top, votes[top] / sum(votes.values())

    def summary(self, keep_threshold: float) -> dict:
        best = self.best()
        p = best[1].p_animal if best else 0.0
        species, share = self.species()
        return {
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "frames": len(self.frames),
            "first": self.names[0] if self.names else None,
            "kept": p >= keep_threshold,
            "p_animal": round(p, 4),
            "species": species if p >= keep_threshold else None,
            "species_share": round(share, 3) if p >= keep_threshold else None,
            "best_frame": self.names[best[0]] if best else None,
            "best_box": best[1].bbox if best else None,
            "boxes": sum(len(f.boxes) for f in self.frames),
        }


class EventGrouper:
    """Feeds frames in time order; returns an Event when a gap longer than ``gap`` closes it."""

    def __init__(self, gap: float = 10.0):
        self.gap = gap
        self.open: Event | None = None

    def add(self, frame: FrameResult, name: str = "") -> Event | None:
        done = None
        if self.open and frame.t - self.open.end > self.gap:
            done, self.open = self.open, None
        if self.open is None:
            self.open = Event()
        self.open.frames.append(frame)
        self.open.names.append(name)
        return done

    def flush(self) -> Event | None:
        done, self.open = self.open, None
        return done


class AnimalSpans:
    """For continuous video, where frames never stop: an event is a run of frames with a likely
    animal (best box at or above ``keep_threshold``), closed after ``gap`` seconds without one.
    Empty frames aren't kept, so an hour of video holds only the animal moments in memory."""

    def __init__(self, keep_threshold: float, gap: float = 3.0):
        self.threshold, self.gap = keep_threshold, gap
        self.open: Event | None = None

    def add(self, frame: FrameResult, name: str = "") -> Event | None:
        done = None
        if self.open and frame.t - self.open.end > self.gap:
            done, self.open = self.open, None
        if frame.p_animal >= self.threshold:
            if self.open is None:
                self.open = Event()
            self.open.frames.append(frame)
            self.open.names.append(name)
        return done

    def flush(self) -> Event | None:
        done, self.open = self.open, None
        return done
