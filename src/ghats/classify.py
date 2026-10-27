"""Helpers for the Phase 2 species classifier that do not need torch.

The crops are very unbalanced (3,949 background crops in train, 123 pangolin),
so the loss is weighted per class and results are reported per class and as
macro F1, where every class counts the same however rare it is.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def class_weights(counts: Sequence[int], power: float = 0.5) -> np.ndarray:
    """Loss weight per class, ``(1 / count) ** power``, scaled so the mean weight is 1.

    power 0 gives equal weights, 1 makes every class contribute equally to the
    loss; in between softens the imbalance without overfitting the rare classes.
    """
    c = np.asarray(counts, dtype=float)
    if (c <= 0).any():
        raise ValueError("every class needs at least one training example")
    w = c**-power
    return w / w.mean()


def per_class_report(y_true: Sequence[int], y_pred: Sequence[int], classes: Sequence[str]) -> pd.DataFrame:
    """Precision, recall, F1 and support per class, plus a final "macro avg" row.

    - precision: of the crops predicted as this class, the share that really are
    - recall: of the crops that really are this class, the share found
    - F1: harmonic mean of the two, low if either is low
    """
    t = np.asarray(y_true)
    p = np.asarray(y_pred)
    rows = []
    for k, name in enumerate(classes):
        tp = int(((p == k) & (t == k)).sum())
        n_pred = int((p == k).sum())
        n_true = int((t == k).sum())
        prec = tp / n_pred if n_pred else 0.0
        rec = tp / n_true if n_true else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if n_true and prec + rec > 0 else (0.0 if n_true else float("nan"))
        rows.append({"class": name, "precision": prec, "recall": rec, "f1": f1, "support": n_true})
    df = pd.DataFrame(rows)
    present = df[df["support"] > 0]
    macro = {
        "class": "macro avg",
        "precision": present["precision"].mean(),
        "recall": present["recall"].mean(),
        "f1": present["f1"].mean(),
        "support": int(df["support"].sum()),
    }
    return pd.concat([df, pd.DataFrame([macro])], ignore_index=True)


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int) -> np.ndarray:
    """Rows are the true class, columns the predicted class."""
    m = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(m, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return m
