"""Phase 3 extra, step 4: does the re-ID model work on leopards from another survey?

    python scripts/19_leopard102_zero_shot.py --config configs/phase3.yaml

The ArcFace model from 14_train_reid.py was trained and tested on Leopard ID 2022
(Botswana). Here it runs unchanged on LeopardID102 (Zimbabwe, 2021): other leopards,
other cameras, other habitat. No training or tuning on this data.

Scored the same way as Phase 3 (ghats.reid.retrieval_metrics): every crop is a query
against all other crops, same-encounter crops removed, same flank only (flank from
18_flank_classifier.py), top-k averaged per leopard. Leopards seen in only one
encounter can't be queries but stay in the gallery as distractors. Also reported:
- the untrained DINOv2-S backbone, to see what fine-tuning on Botswana leopards adds;
- night vs day queries (by capture hour), since many photos are flash or IR at night;
- the same model scored without the flank rule, to show what the flank labels change;
- a cross-camera score, where a match must come from a different camera. Most correct
  matches in the main score come from the query's own camera weeks later: same
  background and angle, which the model can lean on. Cross-camera is the number for
  following a leopard between cameras.
Plus where the correct top-1 matches come from (same camera, same day, time gap), as a
check for near-duplicate leaks.

Writes reports/phase3/leopard102_zero_shot.md. Embeddings cached in
outputs/phase3/leopard102/embeddings.npz.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ghats.config import load_config
from ghats.reid import retrieval_metrics

NIGHT = (18, 6)  # capture hour >= 18 or < 6


def embeddings(paths: list[Path], cfg: dict, model_path: Path, cache: Path) -> dict[str, np.ndarray]:
    if cache.exists():
        z = np.load(cache)
        if len(z["arcface"]) == len(paths):
            return {k: z[k] for k in z.files}
    import torch

    from ghats.reid_model import Embedder, embed, eval_loader, load_trained

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trained = load_trained(model_path, device)
    untrained = Embedder(cfg["backbone"], cfg["embed_dim"]).to(device)
    out = {
        "arcface": embed(trained, eval_loader(paths, cfg, trained), device),
        "untrained": embed(untrained, eval_loader(paths, cfg, untrained), device, raw=True),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


def top1_sources(emb: np.ndarray, crops: pd.DataFrame, times: pd.Series) -> pd.DataFrame:
    """For each query whose top-1 is the right leopard: same camera? same day? hours apart?"""
    ids, enc, view = (crops[c].to_numpy() for c in ("name", "encounter", "viewpoint"))
    cam = crops["camera"].to_numpy()
    sim = emb @ emb.T
    rows = []
    for q in range(len(crops)):
        keep = (enc != enc[q]) & (view == view[q])
        keep[q] = False
        cand = np.flatnonzero(keep)
        if not (ids[cand] == ids[q]).any():
            continue
        m = cand[np.argmax(sim[q, cand])]
        if ids[m] == ids[q]:
            gap = abs(times.iloc[q] - times.iloc[m])
            rows.append(
                {
                    "same_camera": cam[q] == cam[m],
                    "same_day": gap < pd.Timedelta(days=1),
                    "gap_h": gap / pd.Timedelta(hours=1),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase3.yaml")
    ap.add_argument("--model", default="outputs/phase3/reid/best.pt")
    ap.add_argument("--reports", default="reports/phase3")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ks = cfg["reid"]["ks"]
    lp_dir = Path(cfg["leopard102"]["out"])
    crops = pd.read_csv(lp_dir / "crops.csv", keep_default_na=False)
    paths = [lp_dir / "images" / f"{c}.jpg" for c in crops["crop_id"]]
    emb = embeddings(paths, cfg["train"], Path(args.model), Path("outputs/phase3/leopard102/embeddings.npz"))

    times = pd.to_datetime(crops["time"])
    night = ((times.dt.hour >= NIGHT[0]) | (times.dt.hour < NIGHT[1])).to_numpy()
    # Treating every leopard+camera as one "encounter" removes all same-camera matches.
    by_camera = crops["name"] + "@" + crops["camera"]

    def score(name: str, flank: bool = True, queries=None, encounters=None) -> dict:
        views = crops["viewpoint"] if flank else None
        enc = crops["encounter"] if encounters is None else encounters
        return retrieval_metrics(emb[name], crops["name"], enc, views, ks, queries)

    runs = {
        "DINOv2-S, untrained": score("untrained"),
        "**DINOv2-S + ArcFace (trained on Leopard ID 2022)**": score("arcface"),
        "ArcFace, night queries only": score("arcface", queries=night),
        "ArcFace, day queries only": score("arcface", queries=~night),
        "ArcFace, flank rule off (any side can match)": score("arcface", flank=False),
        "DINOv2-S untrained, match must be at another camera": score("untrained", encounters=by_camera),
        "**ArcFace, match must be at another camera**": score("arcface", encounters=by_camera),
    }
    src = top1_sources(emb["arcface"], crops, times)
    cols = [f"top{k}_per_id" for k in ks] + ["mAP", "queries"]
    heads = [f"Top-{k}" for k in ks] + ["mAP", "Queries"]

    def fmt(c: str, v: float) -> str:
        return str(int(v)) if c == "queries" else f"{v:.1%}"

    main_run = runs["**DINOv2-S + ArcFace (trained on Leopard ID 2022)**"]
    query_ids = crops.loc[crops.groupby("name")["encounter"].transform("nunique") >= 2, "name"].nunique()
    p = main_run["top1_per_id"]
    se = np.sqrt(p * (1 - p) / query_ids)
    lines = [
        "# Phase 3 model on LeopardID102 (another survey, no retraining)",
        "",
        (
            f"Generated by `scripts/19_leopard102_zero_shot.py`. Gallery: {len(crops)} crops of "
            f"{crops['name'].nunique()} Zimbabwe leopards; {query_ids} leopards have 2+ encounters and act as "
            "queries. Same-encounter matches removed, same flank only, top-k averaged per leopard. "
            f"Night = capture hour >= {NIGHT[0]}:00 or < {NIGHT[1]}:00 ({night.mean():.0%} of crops)."
        ),
        "",
        "| Model | " + " | ".join(heads) + " |",
        "|---|" + "---|" * len(cols),
        *("| " + name + " | " + " | ".join(fmt(c, r[c]) for c in cols) + " |" for name, r in runs.items()),
        "",
        (
            f"Top-1 per leopard has a standard error of about ±{se:.1%} with {query_ids} query leopards. "
            "For comparison, on its own test set (107 Leopard ID 2022 leopards) the ArcFace model scores "
            "top-1 40.2%, top-5 57.1%, top-10 64.5%, mAP 35.4% (reid_finetune.md)."
        ),
        "",
        "## Where the correct top-1 matches come from (ArcFace, main score)",
        "",
        (
            f"- {len(src)} correct top-1 matches: {src['same_camera'].mean():.0%} from the query's own camera, "
            f"{src['same_day'].mean():.0%} from the same day, {(src['gap_h'] < 2).mean():.0%} within 2 hours; "
            f"median gap {src['gap_h'].median() / 24:.0f} days."
        ),
        (
            "- So near-duplicates from one visit are not what drives the score, but a shared camera view is. "
            "The other-camera rows remove it."
        ),
    ]
    reports = Path(args.reports)
    (reports / "leopard102_zero_shot.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
