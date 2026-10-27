"""Phase 2, step 4: does a crop classifier remove more empty events than the detector alone?

    python scripts/09_event_eval.py --config configs/phase2.yaml

Every val/test trigger event gets one score per method:
  detector            max detector confidence over the event's boxes (the Phase 1 rule)
  <model>             max P(not background) over the event's boxes
  <model> x detector  max of confidence x P(not background): a box must convince both
Events with no box at all score 0. For each method the threshold that keeps 98% of
animal events is picked on val and applied unchanged to test.

Models are our classifier (outputs/phase2/classifier/predictions.csv) and any baseline
that scripts/08_run_baselines.py has scored. Only crops scored by every model are used,
so all methods see exactly the same boxes. Also compares per-class F1 on the classes
every model can name. Writes reports/phase2/event_eval.{md,csv}.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ghats.classify import per_class_report
from ghats.config import load_config
from ghats.crops import BACKGROUND
from ghats.metrics import recall_by_group, threshold_for_recall, threshold_sweep


def load_events(manifest: Path, splits: Path) -> pd.DataFrame:
    m = pd.read_csv(manifest, dtype={"image_id": str})
    ev = m.drop_duplicates("seq_key")[["seq_key", "location", "seq_gt", "seq_group"]]
    split = pd.read_csv(splits).set_index("location")["split"]
    ev = ev.assign(split=ev["location"].map(split))
    ev = ev[ev["split"].isin(["val", "test"]) & ev["seq_gt"].isin(["animal", "empty"])]
    return ev.rename(columns={"seq_gt": "gt", "seq_group": "group"}).reset_index(drop=True)


def event_scores(crops: pd.DataFrame, score: pd.Series, events: pd.DataFrame) -> pd.Series:
    best = score.groupby(crops["seq_key"]).max()
    return events["seq_key"].map(best).fillna(0.0)


def markdown(df: pd.DataFrame, fmt) -> str:
    head = "| " + " | ".join([df.index.name or "", *map(str, df.columns)]) + " |"
    rule = "|" + "---|" * (len(df.columns) + 1)
    body = ["| " + " | ".join([str(i), *(fmt(v) for v in row)]) + " |" for i, row in zip(df.index, df.to_numpy())]
    return "\n".join([head, rule, *body])


def evaluate(events: pd.DataFrame, target: float) -> dict:
    val, test = events[events["split"] == "val"], events[events["split"] == "test"]
    t = threshold_for_recall(val.loc[val["gt"] == "animal", "score"], target)
    row = threshold_sweep(test["score"], test["gt"] == "animal", [t]).iloc[0]
    return {
        "threshold": t,
        "test_recall": row["recall"],
        "empty_removed": row["empty_removed"],
        "review_reduction": row["review_reduction"],
        "animals_missed": int(row["fn"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase2.yaml")
    ap.add_argument("--manifest", default="data/swg/manifest_phase2.csv")
    ap.add_argument("--splits", default="data/crops/splits.csv")
    ap.add_argument("--ours", default="outputs/phase2/classifier/predictions.csv")
    ap.add_argument("--baselines", default="outputs/phase2/baselines")
    ap.add_argument("--reports", default="reports/phase2")
    args = ap.parse_args()

    cfg = load_config(args.config)
    target = cfg["evaluation"]["target_recall"]
    groups = sorted(pd.read_csv(args.ours, nrows=0).filter(regex="^p_").columns.str[2:])

    models = {"ours": pd.read_csv(args.ours)}
    covers = {"ours": set(groups)}
    for name, b in cfg.get("baselines", {}).items():
        path = Path(args.baselines) / name / "predictions.csv"
        if path.exists():
            models[name] = pd.read_csv(path)
            mapped = b["classes"].values() if "classes" in b else (r["group"] for r in b["rules"])
            covers[name] = set(mapped)
        else:
            print(f"(no {path}, skipping {name})")

    # Same boxes for every method.
    shared = set.intersection(*(set(df["crop_id"]) for df in models.values()))
    models = {n: df[df["crop_id"].isin(shared)].set_index("crop_id").sort_index() for n, df in models.items()}
    crops = models["ours"]
    events = load_events(Path(args.manifest), Path(args.splits))
    print(f"{len(shared):,} crops in {crops['seq_key'].nunique():,} events; models: {', '.join(models)}")

    methods = {"detector": crops["conf"]}
    for name, df in models.items():
        p_animal = 1.0 - df[f"p_{BACKGROUND}"]
        methods[name] = p_animal
        methods[f"{name} x detector"] = p_animal * crops["conf"]

    rows, by_group = [], {}
    for name, score in methods.items():
        ev = events.assign(score=event_scores(crops, score, events).to_numpy())
        for t in sorted({target, 0.95, 0.90}, reverse=True):
            rows.append({"target": t, "method": name, **evaluate(ev, t)})
        test = ev[ev["split"] == "test"]
        at_target = next(r for r in rows if r["method"] == name and r["target"] == target)
        by_group[name] = recall_by_group(test, at_target["threshold"]).set_index("group")["recall"]
    table = pd.DataFrame(rows).sort_values(["target", "method"], ascending=[False, True], kind="stable")
    groups_recall = pd.DataFrame(by_group)

    # Per-class F1 on test crops the classifier was meant to handle, over classes every model can name.
    common = sorted(set.intersection(*covers.values()))
    test_ok = crops[(crops["split"] == "test") & crops["train_ok"] & crops["label"].isin(common)]
    names = common + ["(other)"]
    idx = {g: k for k, g in enumerate(common)}
    f1 = {}
    for name, df in models.items():
        sub = df.loc[test_ok.index, [f"p_{g}" for g in common]]
        pred = sub.to_numpy().argmax(1) if name != "ours" else test_ok["pred"].map(idx).fillna(len(common))
        if name != "ours":
            # A baseline's top choice may be a class we do not use: count that as "other".
            other = df.loc[test_ok.index].filter(regex="^p_").drop(columns=sub.columns).max(1).to_numpy()
            pred = np.where(other > sub.max(1).to_numpy(), len(common), pred)
        rep = per_class_report(test_ok["label"].map(idx), np.asarray(pred, dtype=int), names)
        f1[name] = rep.set_index("class")["f1"].drop(["(other)", "macro avg"])
    f1 = pd.DataFrame(f1)

    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    table.to_csv(reports / "event_eval.csv", index=False)
    n_test = events[events["split"] == "test"]
    n_animal, n_empty = (n_test["gt"] == "animal").sum(), (n_test["gt"] == "empty").sum()
    n_cams = n_test["location"].nunique()
    lines = [
        "# Phase 2: empty-event filtering with a crop classifier",
        "",
        f"Test: {len(n_test):,} events ({n_animal:,} animal, {n_empty:,} empty) from {n_cams} cameras not used in training.",
        "Each threshold is the highest that keeps the target share of animal events on val, applied unchanged to test.",
        "",
        "| Val target | Method | Threshold | Test recall | Empty events removed | Review workload removed | Animals missed |",
        "|---|---|---|---|---|---|---|",
        *(
            f"| {r.target:.0%} | {r.method} | {r.threshold:.3f} | {r.test_recall:.1%} | {r.empty_removed:.1%} | "
            f"{r.review_reduction:.1%} | {r.animals_missed} |"
            for r in table.itertuples()
        ),
        "",
        f"## Test recall by species group at the {target:.0%} threshold",
        "",
        markdown(groups_recall, lambda v: f"{v:.0%}"),
        "",
        f"## Crop-level F1 on test, classes every model can name ({len(test_ok):,} crops)",
        "",
        markdown(f1, lambda v: f"{v:.2f}"),
        "",
        "Macro F1: " + ", ".join(f"{n} {v:.3f}" for n, v in f1.mean().items()),
    ]
    (reports / "event_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
