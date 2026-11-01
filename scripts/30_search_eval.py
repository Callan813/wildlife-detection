"""Phase 4, step 2: how well does search find the right trigger events?

    python scripts/30_search_eval.py

Queries whose answers the labels already give, scored on the 566 test-camera events
(val cameras pick the settings):

  species   one query per species group ("sambar"). Ranked by the text embedding against
            each event's best frame, best crop, or either, for CLIP and BioCLIP 2 and three
            kinds of prompt (common name, scientific name, both); by the Phase 2 probe's
            probability; and by probe + embedding.
  empty     "an empty photo with no animals": embedding vs the probe.
  night     "a photo taken at night": embedding vs the colour score (IR shots are grey),
            with the capture hour as the truth.
  filter    the species filter on its own (probe probability >= species_min_prob): what
            share of the events it keeps are the right species, and what share it finds.

Metrics are average precision (AP: 1.0 = every right event ranked above every wrong
one) and precision in the top 10, averaged over the queries.

Questions the labels can't check ("drinking", "close to the camera") are in
scripts/32_search_gallery.py, judged by eye.

Writes reports/phase4/search_eval.{md,csv} and species_ap.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ghats.config import load_config
from ghats.search import average_precision, load_index, precision_at, species_prompts

EMPTY_PROMPTS = ["an empty camera trap photo with no animals.", "a photo of empty forest."]
NIGHT_PROMPTS = ["a camera trap photo taken at night.", "a black and white infrared photo at night."]


def text_encoder(cfg: dict, name: str):
    import open_clip
    import torch

    mdir = Path(cfg["models"]["dir"]) / name
    model, _, _ = open_clip.create_model_and_transforms(f"local-dir:{mdir}")
    tok = open_clip.get_tokenizer(f"local-dir:{mdir}")
    model.eval()

    @torch.no_grad()
    def encode(texts: list[str]) -> np.ndarray:
        f = model.encode_text(tok(texts))
        f = f / f.norm(dim=-1, keepdim=True)
        return f.numpy()

    return encode


def mean_vec(encode, prompts: list[str]) -> np.ndarray:
    v = encode(prompts).mean(axis=0)
    return v / np.linalg.norm(v)


def nice(method: str) -> str:
    """bioclip-2|taxon|crops -> bioclip-2 / taxon / crops (a | would split a markdown table cell)."""
    return method.replace("|", " / ")


def floor(x: np.ndarray) -> np.ndarray:
    """Events with nothing to compare (no crops) rank last instead of scoring -inf."""
    ok = np.isfinite(x)
    return np.where(ok, x, x[ok].min() - 1.0) if ok.any() else np.zeros_like(x)


def zscore(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return (x - x[mask].mean()) / (x[mask].std() + 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase4.yaml")
    ap.add_argument("--reports", default="reports/phase4")
    args = ap.parse_args()
    cfg = load_config(args.config)
    groups = list(cfg["species"])
    out = Path(args.reports)
    out.mkdir(parents=True, exist_ok=True)

    rows, per_group, conf_sweep = [], [], []
    sims: dict[str, dict] = {}  # model -> {(group, style, use): per-event similarity}
    idx = None
    for name in cfg["models"]["names"]:
        idx = load_index(cfg, name)
        encode = text_encoder(cfg, name)
        sims[name] = {}
        for g in groups:
            sp = cfg["species"][g]
            for style in ["common", "taxon", "both"]:
                v = mean_vec(encode, species_prompts(g, sp["words"], sp.get("taxon"), style))
                for use in ["frames", "crops", "both"]:
                    sims[name][(g, style, use)] = idx.similarity(v, use)
        for use in ["frames", "crops", "both"]:
            sims[name][("empty", use)] = idx.similarity(mean_vec(encode, EMPTY_PROMPTS), use)
        sims[name]["night"] = idx.similarity(mean_vec(encode, NIGHT_PROMPTS), "frames")
        if name == cfg["models"]["search"]:
            # Which crops should take part: sweep the detector-confidence floor (taxon prompts, frames + crops).
            vecs = {
                g: mean_vec(
                    encode, species_prompts(g, cfg["species"][g]["words"], cfg["species"][g].get("taxon"), "taxon")
                )
                for g in groups
            }
            keep = idx.crop_min_conf
            for c in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]:
                idx.crop_min_conf = c
                sc = {g: idx.similarity(vecs[g], "both") for g in groups}
                conf_sweep.append((c, sc))
            idx.crop_min_conf = keep

    ev = idx.events
    split = ev["split"].to_numpy()

    def species_scores(method: str) -> dict[str, np.ndarray]:
        """method -> {group: per-event score}."""
        if method == "probe":
            return {g: ev[f"p_{g}"].to_numpy() for g in groups}
        name, style, use = method.split("|")
        return {g: floor(sims[name][(g, style, use)]) for g in groups}

    def evaluate(scores: dict[str, np.ndarray], part: str) -> tuple[float, float, dict[str, float]]:
        m = split == part
        aps = {g: average_precision(scores[g][m], ev["truth"].to_numpy()[m] == g) for g in groups}
        p10 = [precision_at(scores[g][m], ev["truth"].to_numpy()[m] == g, 10) for g in groups]
        return float(np.nanmean(list(aps.values()))), float(np.mean(p10)), aps

    methods = ["probe"] + [
        f"{n}|{style}|{use}"
        for n in cfg["models"]["names"]
        for style in ["common", "taxon", "both"]
        for use in ["frames", "crops", "both"]
    ]
    results = {}
    for meth in methods:
        sc = species_scores(meth)
        results[meth] = sc
        for part in ["val", "test"]:
            mAP, p10, aps = evaluate(sc, part)
            rows.append({"query": "species", "method": meth, "split": part, "mean_AP": mAP, "p_at_10": p10})
            if part == "test":
                per_group.append({"method": meth, **aps})

    # Probe + embedding: z-scored sum, weight picked on val.
    val_rows = [r for r in rows if r["split"] == "val" and r["method"] != "probe"]
    best_emb = max(val_rows, key=lambda r: r["mean_AP"])["method"]
    fusion_val = []
    for w in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
        sc = {}
        for g in groups:
            fused = np.zeros(len(ev))
            for part in ["val", "test"]:
                m = split == part
                fused[m] = zscore(results["probe"][g], m)[m] + w * zscore(results[best_emb][g], m)[m]
            sc[g] = fused
        fusion_val.append((evaluate(sc, "val")[0], w, sc))
    _, w_best, sc = max(fusion_val, key=lambda t: t[0])
    fmeth = f"probe + {w_best} x {best_emb}"
    for part in ["val", "test"]:
        mAP, p10, aps = evaluate(sc, part)
        rows.append({"query": "species", "method": fmeth, "split": part, "mean_AP": mAP, "p_at_10": p10})
    per_group.append({"method": fmeth, **evaluate(sc, "test")[2]})

    sweep_rows = []
    for c, sc in conf_sweep:
        sweep_rows.append({"crop_min_conf": c, "val_AP": evaluate(sc, "val")[0], "test_AP": evaluate(sc, "test")[0]})
    sweep = pd.DataFrame(sweep_rows)
    c_best = float(sweep.loc[sweep["val_AP"].idxmax(), "crop_min_conf"])
    print(f"crop confidence floor: best on val {c_best} (config has {cfg['search'].get('crop_min_conf', 0.0)})")

    # Empty and night.
    truth_empty = (ev["truth"] == "empty").to_numpy()
    a, b = cfg["search"]["night_hours"]
    truth_night = ((ev["hour"] >= a) | (ev["hour"] < b)).to_numpy()
    for part in ["val", "test"]:
        m = split == part
        r = {"split": part}
        rows.append({**r, "query": "empty", "method": "probe (1 - P animal)",
                     "mean_AP": average_precision(-ev["p_animal"].to_numpy()[m], truth_empty[m]),
                     "p_at_10": precision_at(-ev["p_animal"].to_numpy()[m], truth_empty[m], 10)})  # fmt: skip
        for n in cfg["models"]["names"]:
            for use in ["frames", "both"]:
                s = sims[n][("empty", use)][m]
                rows.append({**r, "query": "empty", "method": f"{n}|{use}",
                             "mean_AP": average_precision(s, truth_empty[m]), "p_at_10": precision_at(s, truth_empty[m], 10)})  # fmt: skip
            s = sims[n]["night"][m]
            rows.append({**r, "query": "night", "method": f"{n}|frames",
                         "mean_AP": average_precision(s, truth_night[m]), "p_at_10": precision_at(s, truth_night[m], 10)})  # fmt: skip
        s = -ev["colour"].to_numpy()[m]
        rows.append({**r, "query": "night", "method": "colour score (grey = night)",
                     "mean_AP": average_precision(s, truth_night[m]), "p_at_10": precision_at(s, truth_night[m], 10)})  # fmt: skip
        # A random ranking scores about the share of events that are right.
        share = np.mean([(ev["truth"].to_numpy()[m] == g).mean() for g in groups])
        for qname, base in [("species", share), ("empty", truth_empty[m].mean()), ("night", truth_night[m].mean())]:
            rows.append({**r, "query": qname, "method": "chance (random order)", "mean_AP": base, "p_at_10": base})

    # The species filter: pick the probability cut on val (best mean F1), report test.
    def filter_stats(t: float, part: str) -> tuple[float, float, float]:
        m = split == part
        prec, rec, f1 = [], [], []
        for g in groups:
            keep = (ev[f"p_{g}"].to_numpy() >= t)[m]
            rel = (ev["truth"].to_numpy() == g)[m]
            if not rel.any():
                continue
            pr = (keep & rel).sum() / max(keep.sum(), 1)
            rc = (keep & rel).sum() / rel.sum()
            prec.append(pr), rec.append(rc), f1.append(0 if pr + rc == 0 else 2 * pr * rc / (pr + rc))
        return float(np.mean(prec)), float(np.mean(rec)), float(np.mean(f1))

    cuts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    t_best = max(cuts, key=lambda t: filter_stats(t, "val")[2])
    filt = []
    for part in ["val", "test"]:
        pr, rc, f1 = filter_stats(t_best, part)
        filt.append({"split": part, "min_prob": t_best, "precision": pr, "recall": rc, "F1": f1})
    print(f"species filter: best cut on val {t_best} (config has {cfg['search']['species_min_prob']})")

    df = pd.DataFrame(rows)
    df.to_csv(out / "search_eval.csv", index=False, float_format="%.4f")
    pg = pd.DataFrame(per_group).set_index("method")
    pg.to_csv(out / "species_ap.csv", float_format="%.4f")

    test = df[df["split"] == "test"].drop(columns="split")
    val = df[df["split"] == "val"].set_index(["query", "method"])["mean_AP"]
    test = test.assign(val_AP=[val[(q, m)] for q, m in zip(test["query"], test["method"])])
    n_test = int((split == "test").sum())
    lines = [
        "# Phase 4: search evaluation",
        "",
        f"{n_test} trigger events from the test cameras (val cameras: {int((split == 'val').sum())} events).",
        "AP = average precision over all test events; P@10 = share of the top 10 that are right.",
        "",
    ]
    for qname in ["species", "empty", "night"]:
        t = test[test["query"] == qname].sort_values("val_AP", ascending=False)
        if qname == "species":
            t = pd.concat(
                [t.head(8), t[t["method"].str.startswith("clip-b16")].head(2), t[t["method"].str.startswith("chance")]]
            ).drop_duplicates()
        lines += [f"## {qname}", "", "| Method | test AP | test P@10 | val AP |", "|---|---|---|---|"]
        lines += [f"| {nice(r.method)} | {r.mean_AP:.3f} | {r.p_at_10:.2f} | {r.val_AP:.3f} |" for r in t.itertuples()]
        lines.append("")
    f = pd.DataFrame(filt)
    lines += [
        "## species filter",
        "",
        f"Keep an event when the probe gives the species >= {t_best} (best mean F1 on val).",
        "",
    ]
    lines += ["| Split | precision | recall | F1 |", "|---|---|---|---|"]
    lines += [f"| {r.split} | {r.precision:.3f} | {r.recall:.3f} | {r.F1:.3f} |" for r in f.itertuples()]
    lines += [
        "",
        "## which crops to match",
        "",
        f"Species queries ({cfg['models']['search']}, scientific-name prompts, frames + crops) when only crops",
        f"with detector confidence >= the floor take part. Best on val: {c_best}.",
        "",
        "| Floor | val AP | test AP |",
        "|---|---|---|",
    ]
    lines += [f"| {r.crop_min_conf} | {r.val_AP:.3f} | {r.test_AP:.3f} |" for r in sweep.itertuples()]
    lines += ["", "## species AP by group (test)", ""]
    show = [m for m in ["probe", best_emb, fmeth, "clip-b16|common|both"] if m in pg.index]
    lines += ["| Group | " + " | ".join(nice(m) for m in show) + " |", "|---|" + "---|" * len(show)]
    for g in groups:
        lines.append(f"| {g} | " + " | ".join(f"{pg.loc[m, g]:.3f}" for m in show) + " |")
    (out / "search_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
