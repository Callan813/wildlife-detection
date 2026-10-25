"""Event scoring rules that try to beat "max frame score" at rejecting false alarms.

Each rule turns the animal boxes of one trigger event into a single score, so it
can be swept and compared with the plain max-score baseline. Boxes are
``(conf, [x1, y1, x2, y2])`` with normalised coordinates.

Two ideas are tested in Phase 1, and on SWG neither beats the baseline
(see ``scripts/05_filter_experiments.py``):

- burst agreement: trust a detection only if other frames of the same burst see it too
- per-camera background: ignore boxes at a spot where the same camera fires in other events
"""

from __future__ import annotations

from collections.abc import Sequence

Box = tuple[float, Sequence[float]]


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def max_score(frames: Sequence[Sequence[Box]]) -> float:
    """Baseline: the best animal box anywhere in the event."""
    return max((c for boxes in frames for c, _ in boxes), default=0.0)


def kth_frame_score(frames: Sequence[Sequence[Box]], k: int = 2) -> float:
    """Score at which at least ``k`` frames have an animal box.

    Events shorter than ``k`` frames fall back to the max score, so single-frame
    triggers are not thrown away just for being short.
    """
    per_frame = sorted((max((c for c, _ in boxes), default=0.0) for boxes in frames), reverse=True)
    if len(per_frame) < k:
        return per_frame[0] if per_frame else 0.0
    return per_frame[k - 1]


def spatial_agreement_score(frames: Sequence[Sequence[Box]], k: int = 2, min_iou: float = 0.3) -> float:
    """Like :func:`kth_frame_score`, but the ``k`` frames must see a box in the same place."""
    if len(frames) < k:
        return max_score(frames)
    best = 0.0
    for i, boxes in enumerate(frames):
        for conf, box in boxes:
            support = [conf]
            for j, other in enumerate(frames):
                if j == i:
                    continue
                matched = [c for c, b in other if iou(box, b) >= min_iou]
                if matched:
                    support.append(max(matched))
            if len(support) >= k:
                best = max(best, sorted(support, reverse=True)[k - 1])
    return best


def background_suppressed_score(
    frames: Sequence[Sequence[Box]],
    other_event_boxes: Sequence[Sequence[float]],
    min_iou: float = 0.7,
    min_seen: int = 1,
) -> float:
    """Max score after dropping boxes that sit where this camera also fired in other events.

    ``other_event_boxes`` are boxes from other events at the same camera (no labels
    are used). A box that overlaps at least ``min_seen`` of them is treated as
    background, such as a leaf in front of the lens.
    """
    best = 0.0
    for boxes in frames:
        for conf, box in boxes:
            seen = sum(iou(box, o) >= min_iou for o in other_event_boxes)
            if seen < min_seen:
                best = max(best, conf)
    return best
