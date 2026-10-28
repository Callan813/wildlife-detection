"""Phase 3, step 4: can the system say "this is a leopard we haven't seen"?

    python scripts/15_open_set.py --config configs/phase3.yaml

Closed-set retrieval always returns the closest leopard, even for an animal that is not
in the catalogue. In a real survey new leopards turn up all the time, so each answer
needs a "new individual" option: if the best match is less similar than a threshold,
say "new" instead.

Simulation: in each of --draws rounds, a random half of the split's leopards is removed
from the gallery, so their photos should come back as "new". Known leopards must be
matched correctly and confidently. The threshold that best balances the two
(ghats.reid.open_set_rates), and the strictest one that keeps 90% of the correct
identifications, are picked on val and applied unchanged to test.

Models: the ArcFace fine-tune from scripts/14_train_reid.py and, for comparison, the
best untrained baseline (SpeciesNet features cached by 13_reid_baseline.py). Writes
reports/phase3/open_set.{md,csv,png}.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ghats.config import load_config
from ghats.reid import l2_normalise, open_set_queries, open_set_rates

KEEP = 0.9  # share of closed-set identifications the conservative threshold keeps


def simulate(emb: np.ndarray, crops: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    names = sorted(crops["name"].unique())
    rng = random.Random(seed)
    out = []
    for d in range(draws):
        hidden = set(rng.sample(names, len(names) // 2))
        q = open_set_queries(emb, crops["name"], crops["encounter"], crops["viewpoint"], hidden)
        out.append(q.assign(draw=d))
    return pd.concat(out, ignore_index=True)


def rates(queries: pd.DataFrame, t: float) -> dict[str, float]:
    """open_set_rates per draw, averaged over draws."""
    per = pd.DataFrame([open_set_rates(g.reset_index(drop=True), t) for _, g in queries.groupby("draw")])
    return per.mean().to_dict()


def auroc(queries: pd.DataFrame) -> float:
    """How well best_sim separates known from new queries (1 = perfectly, 0.5 = chance)."""
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(~queries["is_new"], queries["best_sim"]))


def embeddings(name: str, crops: pd.DataFrame, cfg: dict, args) -> np.ndarray:
    if name == "speciesnet":
        z = np.load(Path(args.baseline_cache) / "speciesnet.npz", allow_pickle=True)
        return l2_normalise(pd.DataFrame(z["x"], index=z["crop_id"]).loc[crops["crop_id"]].to_numpy())
    cache = Path(args.model).with_name("embeddings.npz")
    if cache.exists() and list(np.load(cache)["crop_id"]) == crops["crop_id"].tolist():
        return np.load(cache)["x"]
    import torch

    from ghats.reid_model import embed, eval_loader, load_trained

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_trained(Path(args.model), device)
    out_dir = Path(cfg["crops"]["out"])
    paths = [out_dir / "images" / f"{c}.jpg" for c in crops["crop_id"]]
    emb = embed(model, eval_loader(paths, cfg["train"], model), device)
    np.savez(cache, x=emb, crop_id=crops["crop_id"].to_numpy())
    return emb


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase3.yaml")
    ap.add_argument("--model", default="outputs/phase3/reid/best.pt")
    ap.add_argument("--baseline-cache", default="outputs/phase3/embeddings")
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--reports", default="reports/phase3")
    args = ap.parse_args()

    cfg = load_config(args.config)
    crops = pd.read_csv(Path(cfg["crops"]["out"]) / "crops.csv")
    crops = crops[crops["split"].isin(["val", "test"])].reset_index(drop=True)
    grid = np.round(np.arange(-0.2, 1.0, 0.01), 2)

    rows, curves, summary = [], {}, []
    for model, label in [("arcface", "DINOv2-S + ArcFace"), ("speciesnet", "SpeciesNet, untrained")]:
        emb = embeddings(model, crops, cfg, args)
        q = {}
        for split in ("val", "test"):
            m = (crops["split"] == split).to_numpy()
            q[split] = simulate(emb[m], crops[m].reset_index(drop=True), args.draws, seed=0)
        val_curve = pd.DataFrame([{"t": t, **rates(q["val"], t)} for t in grid])
        val_closed = rates(q["val"], -2.0)["known_identified"]
        keep = val_curve[val_curve["known_identified"] >= KEEP * val_closed]
        rules = {
            "best balance": float(val_curve.loc[val_curve["balanced"].idxmax(), "t"]),
            f"keep {KEEP:.0%} of identifications": float(keep["t"].max()),
        }
        curves[label] = pd.DataFrame([{"t": t, **rates(q["test"], t)} for t in grid])
        summary.append(
            {
                "model": label,
                "known_identified_closed_set": rates(q["test"], -2.0)["known_identified"],
                "auroc_known_vs_new": auroc(q["test"]),
                "new_share_of_queries": float(q["test"]["is_new"].mean()),
            }
        )
        for rule, t in rules.items():
            test = rates(q["test"], t)
            rows.append({"model": label, "rule": rule, "threshold": t, **test})
            print(f"{label}, {rule}: threshold {t:.2f}, test " + ", ".join(f"{k} {v:.3f}" for k, v in test.items()))

    table, summary = pd.DataFrame(rows), pd.DataFrame(summary)
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    table.merge(summary, on="model").to_csv(reports / "open_set.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)
    for label, c in curves.items():
        line = ax.plot(c["new_flagged"] * 100, c["known_identified"] * 100, lw=1.8, label=label)[0]
        pts = table[table["model"] == label]
        for (_, r), marker in zip(pts.iterrows(), ["o", "s"]):
            ax.scatter(
                r["new_flagged"] * 100,
                r["known_identified"] * 100,
                s=50,
                marker=marker,
                zorder=5,
                color=line.get_color(),
                edgecolors="black",
            )
    ax.set_xlabel("New leopards correctly flagged as new (%)")
    ax.set_ylabel("Known leopards correctly identified (%)")
    ax.set_title("Open-set re-ID, unseen test leopards")
    ax.text(
        0.01,
        0.02,
        f"thresholds picked on val: o best balance, □ keep {KEEP:.0%} of identifications",
        transform=ax.transAxes,
        fontsize=7.5,
    )
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(reports / "open_set.png")
    plt.close(fig)

    n_test = crops[crops["split"] == "test"]["name"].nunique()
    lines = [
        "# Phase 3: recognising new leopards (open-set re-ID)",
        "",
        (
            f"Test: {n_test} unseen leopards. In each of {args.draws} draws a random half is removed from the "
            'gallery, so their photos should be answered "new"; photos of leopards seen in only one encounter '
            'count as new too. A photo is answered "new" when its best match is below a threshold picked on val '
            "and applied unchanged to test. Rates are averaged per leopard, then over draws."
        ),
        "",
        (
            "Two ways to pick the threshold: the best balance of the two rates, or the strictest threshold that "
            f'still keeps {KEEP:.0%} of the correct identifications you\'d get with no "new" option.'
        ),
        "",
        "| Model | Threshold rule | Threshold | Known identified | New flagged |",
        "|---|---|---|---|---|",
        *(
            f"| {r.model} | {r.rule} | {r.threshold:.2f} | {r.known_identified:.1%} | {r.new_flagged:.1%} |"
            for r in table.itertuples()
        ),
        "",
        '| Model | Known identified, no "new" option | AUROC known vs new |',
        "|---|---|---|",
        *(
            f"| {r.model} | {r.known_identified_closed_set:.1%} | {r.auroc_known_vs_new:.3f} |"
            for r in summary.itertuples()
        ),
        "",
        "![open-set trade-off](open_set.png)",
    ]
    (reports / "open_set.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
