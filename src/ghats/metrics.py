"""Metrics for "does this frame / event contain an animal?" filtering."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def threshold_sweep(scores: Iterable[float], is_animal: Iterable[bool], thresholds: Iterable[float]) -> pd.DataFrame:
    """Confusion counts and filtering metrics at each threshold.

    A unit (frame or event) is *kept* when its score >= threshold.

    - recall: share of animal units kept (animals we did not throw away)
    - precision: share of kept units that really contain an animal
    - empty_removed: share of empty units filtered out
    - review_reduction: share of all units a human no longer has to look at
    """
    s = np.asarray(list(scores), dtype=float)
    y = np.asarray(list(is_animal), dtype=bool)
    if s.shape != y.shape:
        raise ValueError("scores and is_animal must have the same length")

    rows = []
    n = len(s)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    for t in sorted({float(t) for t in thresholds}):
        keep = s >= t
        tp = int((keep & y).sum())
        fp = int((keep & ~y).sum())
        fn = n_pos - tp
        tn = n_neg - fp
        rows.append(
            {
                "threshold": t,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "recall": _div(tp, n_pos),
                "precision": _div(tp, tp + fp),
                "empty_removed": _div(tn, n_neg),
                "review_reduction": _div(n - int(keep.sum()), n),
            }
        )
    return pd.DataFrame(rows)


def pick_operating_point(sweep: pd.DataFrame, target_recall: float) -> pd.Series:
    """Highest threshold whose recall still meets the target.

    Falls back to the threshold with the best recall if none meets it.
    """
    ok = sweep[sweep["recall"] >= target_recall]
    if ok.empty:
        return sweep.sort_values(["recall", "threshold"], ascending=[False, False]).iloc[0]
    return ok.sort_values("threshold").iloc[-1]


def recall_by_group(units: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Recall per species group among animal units at one threshold.

    Expects columns: gt, group, score. Sorted worst-first, because the groups a
    detector misses (small, nocturnal, camouflaged) are the interesting finding.
    """
    animals = units[units["gt"] == "animal"].copy()
    if animals.empty:
        return pd.DataFrame(columns=["group", "n", "recall"])
    animals["kept"] = animals["score"] >= threshold
    out = animals.groupby("group")["kept"].agg(n="size", recall="mean").reset_index()
    return out.sort_values(["recall", "n"], ascending=[True, False]).reset_index(drop=True)


def _div(a: float, b: float) -> float:
    return float(a) / float(b) if b else float("nan")
