"""Turn detector boxes into classifier training crops, and split them by camera.

Phase 2 trains a species classifier on MegaDetector crops. Two things matter:

* Labels. LILA labels whole sequences, not boxes. A box in an empty event is
  background (leaves, rocks, glare that fooled the detector). A box in an animal
  event is that species only if the detector is fairly sure; weak boxes in an
  animal event may be a leaf next to the animal, so they are kept for
  evaluation but not used for training.
* Split. Frames from one camera share a background, so a random split lets the
  model recognise the place instead of the animal. Whole camera locations go to
  train, val or test.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pandas only at runtime where needed: the detector (and so a camera) imports this module
    import pandas as pd

BACKGROUND = "background"
SPLITS = ("train", "val", "test")
# A box this thin (frame pixels) lying on the frame edge is a letterbox-padding phantom.
# Boxes on the padding clamp to zero-height strips, but ones that end a pixel or two inside
# the frame survive as full-width strips along the top or bottom edge, conf <= 0.17. In the
# Phase 2 detections every animal box under 16 px is one of those.
MIN_BOX_PX = 16


def crop_box(bbox: Sequence[float], w: int, h: int, pad: float = 0.1, min_side: int = 32) -> tuple[int, int, int, int]:
    """Normalised [x1, y1, x2, y2] -> padded pixel box (left, top, right, bottom) inside the image.

    The box grows by ``pad`` of its size on every side, and to at least
    ``min_side`` pixels, so the classifier sees a bit of context around small animals.
    """
    x1, y1, x2, y2 = bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h
    bw, bh = x2 - x1, y2 - y1
    grow_x = max(bw * pad, (min_side - bw) / 2, 0)
    grow_y = max(bh * pad, (min_side - bh) / 2, 0)
    left, right = max(0, round(x1 - grow_x)), min(w, round(x2 + grow_x))
    top, bottom = max(0, round(y1 - grow_y)), min(h, round(y2 + grow_y))
    return left, top, max(right, left + 1), max(bottom, top + 1)


def is_padding_strip(bbox: Sequence[float], w: int, h: int) -> bool:
    """Normalised [x1, y1, x2, y2] box thinner than MIN_BOX_PX and touching the frame edge
    across that thin side (top/bottom for wide frames, left/right for tall ones). A tiny
    animal in the open is kept."""
    x1, y1, x2, y2 = bbox
    flat = (y2 - y1) * h < MIN_BOX_PX and (y1 <= 0.0 or y2 >= 1.0)
    narrow = (x2 - x1) * w < MIN_BOX_PX and (x1 <= 0.0 or x2 >= 1.0)
    return flat or narrow


def label_boxes(boxes: pd.DataFrame, min_conf: float, train_min_conf: float) -> pd.DataFrame:
    """Label one row per animal box.

    ``boxes`` needs columns seq_gt (animal | empty), seq_group and conf. Boxes
    below ``min_conf`` are dropped (the detector would filter them anyway).
    Adds ``label`` and ``train_ok``: background boxes are always trainable,
    animal boxes only at ``train_min_conf`` or above.
    """
    out = boxes[boxes["conf"] >= min_conf].copy()
    empty = out["seq_gt"] == "empty"
    out["label"] = out["seq_group"].where(~empty, BACKGROUND)
    out["train_ok"] = empty | (out["conf"] >= train_min_conf)
    return out


def split_locations(
    events: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42, tries: int = 200
) -> dict[str, str]:
    """Assign every camera location to train, val or test.

    ``events`` has one row per trigger event with columns location and label.
    Locations are shuffled and filled into test, then val, then train by event
    count. Some species live at only a few cameras, so several shuffles are tried
    and the one where the worst-covered label is closest to its target share in
    both val and test wins.
    """
    locs = sorted(events["location"].unique())
    per_loc = events.groupby("location").size()
    counts = events.groupby(["label", "location"]).size().unstack(fill_value=0)
    total = len(events)
    rng = random.Random(seed)

    best, best_score = {}, -1.0
    for _ in range(tries):
        order = locs[:]
        rng.shuffle(order)
        assign, filled = {}, {"test": 0, "val": 0}
        for loc in order:
            if filled["test"] < test_frac * total:
                split = "test"
            elif filled["val"] < val_frac * total:
                split = "val"
            else:
                split = "train"
            assign[loc] = split
            if split != "train":
                filled[split] += per_loc[loc]
        score = _coverage(counts, assign, val_frac, test_frac)
        if score > best_score:
            best, best_score = assign, score
    return best


def _coverage(counts: pd.DataFrame, assign: dict[str, str], val_frac: float, test_frac: float) -> float:
    """Worst ratio (actual share / target share) over every label in val and test, capped at 1."""
    import pandas as pd

    split_of = pd.Series(assign).reindex(counts.columns)
    worst = 1.0
    for split, frac in (("val", val_frac), ("test", test_frac)):
        if frac <= 0:
            continue
        share = counts.loc[:, split_of == split].sum(axis=1) / counts.sum(axis=1)
        worst = min(worst, float((share / frac).clip(upper=1).min()))
    return worst
