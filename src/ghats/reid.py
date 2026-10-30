"""Evaluation helpers for individual re-identification (Phase 3).

Re-ID here is retrieval: embed every leopard crop, and for each query crop rank
all other crops by similarity. A good model puts a photo of the same leopard at
the top. Two things make the naive number too optimistic:

* Encounters. Photos from one sighting are near-duplicates (same pose, light,
  background), so matching them is easy and says little about recognising the
  animal weeks later. Gallery crops from the query's own encounter are removed.
* Viewpoint. The left and right flanks carry different spot patterns, so a left
  view can only be matched by another left view. Queries with no possible match
  (a same-leopard, same-viewpoint crop from another encounter) are skipped.

Individuals are split into train / val / test as a whole, so test leopards are
never seen during training.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence

import numpy as np
import pandas as pd


def split_individuals(
    names: pd.Series, val_frac: float = 0.15, test_frac: float = 0.3, seed: int = 42
) -> dict[str, str]:
    """Assign every individual to train, val or test, aiming for the given share of crops."""
    counts = names.value_counts()
    order = list(counts.index)
    random.Random(seed).shuffle(order)
    total = counts.sum()
    out, taken = {}, {"test": 0, "val": 0}
    for name in order:
        if taken["test"] < test_frac * total:
            split = "test"
        elif taken["val"] < val_frac * total:
            split = "val"
        else:
            split = "train"
        out[name] = split
        if split in taken:
            taken[split] += counts[name]
    return out


def assign_encounters(image_ids: pd.Series, names: pd.Series) -> pd.Series:
    """Encounter id per crop: a run of consecutive images (in id order) showing the same leopard.

    For data with no timestamps but ids in capture order. An image with two leopards
    belongs to a run for each. Runs can over-split (another leopard photographed
    mid-sighting), which leaves near-duplicates in the gallery and flatters the
    score, so check how often top-1 matches come from nearby images.
    """
    df = pd.DataFrame({"image_id": image_ids, "name": names})
    order = df.sort_values(["image_id", "name"])
    per_image = order.groupby("image_id")["name"].apply(set)
    ids = per_image.index.to_list()
    run_of: dict[tuple[int, str], int] = {}
    runs = 0
    prev: set[str] = set()
    for i, img in enumerate(ids):
        for name in sorted(per_image[img]):
            if name in prev and i > 0:
                run_of[(img, name)] = run_of[(ids[i - 1], name)]
            else:
                runs += 1
                run_of[(img, name)] = runs
        prev = per_image[img]
    return pd.Series([f"enc{run_of[(i, n)]}" for i, n in zip(image_ids, names)], index=image_ids.index)


_SPOTTED_NAME = re.compile(r"_(CAM\d+)-.*?_(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})_?(.*)$")


def parse_spotted_filename(stem: str) -> dict[str, str]:
    """Camera, capture time and pair side from a LeopardID102 file name (no extension).

    Names look like CTS2021_CAM15-C07-2505_2021-05-11_13-37-30_I_00005a. A trailing
    a/b marks the two cameras of a paired station firing together; "" otherwise.
    """
    m = _SPOTTED_NAME.search(stem)
    if not m:
        raise ValueError(f"unexpected file name: {stem}")
    camera, day, hh, mm, ss, tail = m.groups()
    side = tail[-1] if tail[-1:] in ("a", "b") and tail[-2:-1].isdigit() else ""
    return {"camera": camera, "time": f"{day} {hh}:{mm}:{ss}", "pair_side": side}


def time_encounters(names: pd.Series, cameras: pd.Series, times: pd.Series, gap_minutes: float = 30) -> pd.Series:
    """Encounter id per crop when capture times are known: one leopard at one camera, no gap over gap_minutes.

    Both cameras of a paired station share a camera id, so the a and b frames of one
    pass land in the same encounter.
    """
    df = pd.DataFrame({"name": names, "camera": cameras, "t": pd.to_datetime(times)})
    df = df.sort_values(["name", "camera", "t"])
    new = (
        (df["name"] != df["name"].shift())
        | (df["camera"] != df["camera"].shift())
        | (df["t"] - df["t"].shift() > pd.Timedelta(minutes=gap_minutes))
    )
    return ("enc" + new.cumsum().astype(str)).reindex(names.index)


def retrieval_metrics(
    emb: np.ndarray,
    ids: Sequence[str],
    encounters: Sequence[str],
    views: Sequence[str] | None = None,
    ks: Sequence[int] = (1, 5),
) -> dict[str, float]:
    """Top-k accuracy and mAP over all queries with at least one valid match.

    ``emb`` rows are L2-normalised embeddings. A gallery crop is valid for a query if
    it is from another encounter (and, when ``views`` is given, the same viewpoint).
    Same-encounter crops are removed from the ranking entirely.

    ``top{k}`` averages over queries, so a leopard photographed 150 times counts 150
    times; ``top{k}_per_id`` averages each individual's hit rate first, so every
    individual counts the same (like macro F1).
    """
    ids, enc = np.asarray(ids), np.asarray(encounters)
    v = np.asarray(views) if views is not None else np.zeros(len(ids), dtype=object)
    sim = emb @ emb.T
    hits = {k: [] for k in ks}
    aps, who = [], []
    for q in range(len(ids)):
        keep = (enc != enc[q]) & (v == v[q])
        keep[q] = False
        if not keep.any():
            continue
        same = ids[keep] == ids[q]
        if not same.any():
            continue
        order = np.argsort(-sim[q, keep], kind="stable")
        rel = same[order]
        first = int(np.argmax(rel))
        for k in ks:
            hits[k].append(first < k)
        ranks = np.flatnonzero(rel) + 1
        aps.append(float(np.mean(np.arange(1, len(ranks) + 1) / ranks)))
        who.append(ids[q])
    out = {f"top{k}": float(np.mean(h)) if h else float("nan") for k, h in hits.items()}
    for k, h in hits.items():
        per_id = pd.Series(h, index=who, dtype=float).groupby(level=0).mean()
        out[f"top{k}_per_id"] = float(per_id.mean()) if h else float("nan")
    out["mAP"] = float(np.mean(aps)) if aps else float("nan")
    out["queries"] = len(aps)
    return out


def l2_normalise(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def open_set_queries(
    emb: np.ndarray,
    ids: Sequence[str],
    encounters: Sequence[str],
    views: Sequence[str],
    hidden: set[str],
) -> pd.DataFrame:
    """One row per query when the leopards in ``hidden`` are missing from the gallery.

    The gallery is every crop of a leopard not in ``hidden``; as in retrieval_metrics, a
    query never sees its own encounter and only same-flank crops count. Columns:
      id         the query's leopard
      is_new     True if the gallery holds no valid match (hidden leopard, or a leopard
                 seen in only one encounter): the right answer is "new individual"
      best_sim   similarity of the closest gallery crop
      top1_ok    the closest gallery crop is the same leopard
    A query is answered "new" when best_sim falls below a threshold (see open_set_rates).
    """
    ids, enc, v = np.asarray(ids), np.asarray(encounters), np.asarray(views)
    in_gallery = ~np.isin(ids, list(hidden))
    sim = emb @ emb.T
    rows = []
    for q in range(len(ids)):
        keep = in_gallery & (enc != enc[q]) & (v == v[q])
        keep[q] = False
        if not keep.any():
            continue
        cand = np.flatnonzero(keep)
        top = cand[np.argmax(sim[q, cand])]
        rows.append(
            {
                "id": ids[q],
                "is_new": not (ids[cand] == ids[q]).any(),
                "best_sim": float(sim[q, top]),
                "top1_ok": bool(ids[top] == ids[q]),
            }
        )
    return pd.DataFrame(rows)


def open_set_rates(queries: pd.DataFrame, threshold: float) -> dict[str, float]:
    """At one threshold, averaged per leopard (then over leopards):

    known_identified  known-leopard queries answered with the right leopard
                      (closest match correct and similar enough to be accepted)
    new_flagged       new-leopard queries answered "new" (best match below threshold)
    balanced          the mean of the two
    """
    accept = queries["best_sim"] >= threshold
    known = queries[~queries["is_new"]]
    new = queries[queries["is_new"]]
    k = (known["top1_ok"] & accept[known.index]).groupby(known["id"]).mean().mean()
    n = (~accept[new.index]).groupby(new["id"]).mean().mean()
    return {"known_identified": float(k), "new_flagged": float(n), "balanced": float((k + n) / 2)}
