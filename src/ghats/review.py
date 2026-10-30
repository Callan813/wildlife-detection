"""Review queue helpers for the Phase 5 dashboard.

The dashboard (dashboard/) shows reviewers what the models think and records what the
reviewers decide. These helpers turn model outputs into queue items; the export script
(scripts/22_export_review.py) writes them to the SQLite file the dashboard reads.

Uncertain items go first: a reviewer's time is best spent where the model is least sure.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def audit_ranks(n: int, frac: float, rng: np.random.Generator) -> list[int | None]:
    """Pick a random audit sample: a share of n items gets a rank 1..k in random order, the rest None.

    Uncertain-first review only ever shows items near the decision boundary, so measuring
    the model on them is biased. The audit items are mixed into the queue at random and
    used for accuracy and threshold calibration.
    """
    picked = rng.permutation(n)[: round(n * frac)]
    ranks: list[int | None] = [None] * n
    for r, i in enumerate(picked, start=1):
        ranks[i] = int(r)
    return ranks


def species_uncertainty(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(top-1 index, margin between the top two probabilities) per row. Small margin = unsure."""
    order = np.argsort(-probs, axis=1)
    rows = np.arange(len(probs))
    return order[:, 0], probs[rows, order[:, 0]] - probs[rows, order[:, 1]]


def split_catalogue(crops: pd.DataFrame, known_frac: float = 0.5, seed: int = 0) -> pd.Series:
    """Simulate a survey in progress: which crops start in the catalogue.

    A random share of the leopards is already known, each from its first encounter (by
    lowest image id). Everything else is to be reviewed: later sightings of known
    leopards, and every photo of the leopards nobody has catalogued yet.
    """
    names = np.array(sorted(crops["name"].unique()))
    rng = np.random.default_rng(seed)
    known = set(rng.choice(names, size=round(len(names) * known_frac), replace=False))
    first = crops.groupby("encounter")["image_id"].min().rename("first_image")
    enc_order = crops[["name", "encounter"]].drop_duplicates().join(first, on="encounter")
    first_enc = enc_order.sort_values("first_image").groupby("name")["encounter"].first()
    return crops["name"].isin(known) & (crops["encounter"] == crops["name"].map(first_enc))


def shortlists(
    query_emb: np.ndarray,
    query_views: Sequence[str],
    cat_emb: np.ndarray,
    cat_names: Sequence[str],
    cat_views: Sequence[str],
    cat_ids: Sequence[str],
    k: int = 5,
) -> list[list[dict]]:
    """Top-k catalogued leopards per query, same flank only.

    A leopard's score is its best-matching catalogue crop; that crop is returned as the
    example to show the reviewer side by side. Empty list if no catalogue crop has the
    query's flank.
    """
    cat_names, cat_views, cat_ids = (np.asarray(a) for a in (cat_names, cat_views, cat_ids))
    sim = query_emb @ cat_emb.T
    out = []
    for q, view in enumerate(query_views):
        idx = np.flatnonzero(cat_views == view)
        if not len(idx):
            out.append([])
            continue
        df = pd.DataFrame({"name": cat_names[idx], "sim": sim[q, idx], "crop": cat_ids[idx]})
        best = df.sort_values("sim", ascending=False).drop_duplicates("name").head(k)
        out.append(
            [
                {"rank": r + 1, "leopard": row.name, "sim": float(row.sim), "example_crop_id": str(row.crop)}
                for r, row in enumerate(best.itertuples(index=False))
            ]
        )
    return out
