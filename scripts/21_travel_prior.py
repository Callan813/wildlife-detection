"""Phase 3 extra, step 6: use where and when photos were taken to rule out wrong matches.

    python scripts/21_travel_prior.py --config configs/phase3.yaml

Appearance alone ranks every leopard in the catalogue. LeopardID102 also has capture
times and camera positions (km), so some candidates can be ruled out before looking at
spots. Two rules, applied by sending ruled-out candidates to the bottom of the ranking
(ghats.reid.retrieval_metrics' sim_offset):

- Speed: the leopard would have had to travel between the two cameras faster than
  `max_speed_kmh`, averaged over the time between photos. Set loose, not tuned.
- Distance: the two cameras are more than D km apart. Leopards keep to home ranges, so
  far-apart matches are unlikely, but a hard cutoff will miss a leopard that moves
  further than usual. D has to come from somewhere other than the leopards being scored:
  the query leopards are split in half, D is picked on one half and scored on the other,
  then swapped; repeated over `folds_repeats` random splits and averaged.

Uses the ArcFace embeddings from 19_leopard102_zero_shot.py (model never trained on
these leopards). Scored both ways from step 19: the standard rules, and with the match
forced to come from another camera. Writes reports/phase3/travel_prior.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ghats.config import load_config
from ghats.reid import retrieval_metrics


def camera_distances(crops: pd.DataFrame, locations: Path) -> np.ndarray:
    loc = pd.read_csv(locations).set_index("site")
    xy = loc.loc[crops["camera"], ["x", "y"]].to_numpy()
    return np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])


def rule_out(mask: np.ndarray) -> np.ndarray:
    return np.where(mask, -np.inf, 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase3.yaml")
    ap.add_argument("--reports", default="reports/phase3")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ks = cfg["reid"]["ks"]
    lp = cfg["leopard102"]
    tr = lp["travel"]
    crops = pd.read_csv(Path(lp["out"]) / "crops.csv", keep_default_na=False)
    emb = np.load("outputs/phase3/leopard102/embeddings.npz")["arcface"]

    dist = camera_distances(crops, Path(lp["locations"]))
    t = pd.to_datetime(crops["time"]).to_numpy()
    gap_h = np.abs(t[:, None] - t[None, :]) / np.timedelta64(1, "h")
    too_fast = dist > tr["max_speed_kmh"] * np.maximum(gap_h, 1 / 60)

    names = crops["name"].to_numpy()
    queryable = crops.groupby("name")["encounter"].transform("nunique").to_numpy() >= 2
    query_names = np.unique(names[queryable])
    by_camera = crops["name"] + "@" + crops["camera"]

    def score(offset=None, queries=None, encounters=None) -> dict:
        enc = crops["encounter"] if encounters is None else encounters
        return retrieval_metrics(emb, names, enc, crops["viewpoint"], ks, queries, offset)

    # How many candidates does each rule remove, and how many true matches with them?
    same = names[:, None] == names[None, :]
    off_diag = ~np.eye(len(names), dtype=bool)
    removed = {"speed": too_fast}
    for d in tr["distance_grid_km"]:
        removed[f"distance > {d} km"] = dist > d

    def split_scores(rule_for_half, encounters=None) -> dict:
        """Average metrics over random leopard halves; the rule for each half comes from the other half."""
        rng = np.random.default_rng(0)
        runs, chosen = [], []
        for _ in range(tr["folds_repeats"]):
            perm = rng.permutation(query_names)
            halves = [set(perm[: len(perm) // 2]), set(perm[len(perm) // 2 :])]
            for tune, held in (halves, halves[::-1]):
                d = rule_for_half(np.isin(names, list(tune)), encounters)
                chosen.append(d)
                offset = rule_out((dist > d) | too_fast)
                runs.append(score(offset, np.isin(names, list(held)), encounters))
        out = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
        out["chosen"] = pd.Series(chosen).value_counts().sort_index().to_dict()
        return out

    def best_distance(tune_mask: np.ndarray, encounters) -> float:
        res = {
            d: score(rule_out((dist > d) | too_fast), tune_mask, encounters)["top1_per_id"]
            for d in tr["distance_grid_km"]
        }
        return max(res, key=res.get)

    def same_halves(offset, encounters=None) -> dict:
        """The same random halves as split_scores, scored with a fixed offset, so rows are comparable."""
        rng = np.random.default_rng(0)
        runs = []
        for _ in range(tr["folds_repeats"]):
            perm = rng.permutation(query_names)
            halves = [set(perm[: len(perm) // 2]), set(perm[len(perm) // 2 :])]
            for held in halves[::-1]:
                runs.append(score(offset, np.isin(names, list(held)), encounters))
        return {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}

    rows = {}
    for label, enc in (("standard", None), ("other camera only", by_camera)):
        rows[(label, "appearance only")] = same_halves(None, enc)
        rows[(label, f"+ speed rule ({tr['max_speed_kmh']} km/h)")] = same_halves(rule_out(too_fast), enc)
        rows[(label, "+ speed + distance rule (D picked on other half)")] = split_scores(best_distance, enc)

    cols = [f"top{k}_per_id" for k in ks] + ["mAP"]
    lines = [
        "# Travel prior on LeopardID102",
        "",
        (
            "Generated by `scripts/21_travel_prior.py`. ArcFace model trained on Leopard ID 2022 only. "
            f"Scores are averaged over {tr['folds_repeats']} random splits of the {len(query_names)} query "
            "leopards into halves, each half scored once, so every row uses the same queries."
        ),
        "",
        "## What each rule removes (all pairs of crops)",
        "",
        "| Rule | Candidates removed | True matches removed |",
        "|---|---|---|",
        *(
            f"| {k} | {(m & off_diag).sum() / off_diag.sum():.1%} | {(m & same & off_diag).sum()} of "
            f"{(same & off_diag).sum()} |"
            for k, m in removed.items()
        ),
        "",
        "## Retrieval",
        "",
        "| Scoring | Ranking | " + " | ".join(f"Top-{k}" for k in ks) + " | mAP |",
        "|---|---|" + "---|" * len(cols),
        *(f"| {s} | {r} | " + " | ".join(f"{m[c]:.1%}" for c in cols) + " |" for (s, r), m in rows.items()),
        "",
        "Distance cutoff chosen on the tuning half (count over splits):",
        "",
        *(f"- {s}: {m['chosen']}" for (s, r), m in rows.items() if "chosen" in m),
    ]
    Path(args.reports, "travel_prior.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
