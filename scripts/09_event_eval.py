"""Phase 2, step 4: does a crop classifier remove more empty events than the detector alone?

    python scripts/09_event_eval.py --config configs/phase2.yaml

Every box gets one score per method:
  detector            detector confidence (the Phase 1 rule)
  <model>             P(not background) from a crop classifier
  <model> x detector  confidence x P(not background): a box must convince both
and every val/test trigger event takes either its best box score or its second-best
(an empty event full of leaves is then not kept alive by its single worst leaf; events
with one box keep that box's score). Events with no box score 0. For each method the
threshold that keeps the target share of animal events is picked on val and applied
unchanged to test.

Models are our classifiers (--ours name=predictions.csv, several allowed) and any baseline
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
from ghats.events import kth_box_score
from ghats.metrics import recall_by_group, threshold_for_recall, threshold_sweep


def load_events(manifest: Path, splits: Path) -> pd.DataFrame:
    m = pd.read_csv(manifest, dtype={"image_id": str})
    ev = m.drop_duplicates("seq_key")[["seq_key", "location", "seq_gt", "seq_group"]]
    split = pd.read_csv(splits).set_index("location")["split"]
    ev = ev.assign(split=ev["location"].map(split))
    ev = ev[ev["split"].isin(["val", "test"]) & ev["seq_gt"].isin(["animal", "empty"])]
    return ev.rename(columns={"seq_gt": "gt", "seq_group": "group"}).reset_index(drop=True)


AGGREGATIONS = {"best box": 1, "2nd box": 2}


def event_scores(crops: pd.DataFrame, score: pd.Series, events: pd.DataFrame, k: int) -> pd.Series:
    per_event = kth_box_score(score, crops["seq_key"], k)
    return events["seq_key"].map(per_event).fillna(0.0)


def markdown(df: pd.DataFrame, fmt) -> str:
    head = "| " + " | ".join([df.index.name or "", *map(str, df.columns)]) + " |"
    rule = "|" + "---|" * (len(df.columns) + 1)
    body = ["| " + " | ".join([str(i), *(fmt(v) for v in row)]) + " |" for i, row in zip(df.index, df.to_numpy())]
    return "\n".join([head, rule, *body])


def evaluate(events: pd.DataFrame, target: float) -> dict:
    val, test = events[events["split"] == "val"], events[events["split"] == "test"]
    val_animal = val.loc[val["gt"] == "animal", "score"]
    t = threshold_for_recall(val_animal, target)
    row = threshold_sweep(test["score"], test["gt"] == "animal", [t]).iloc[0]
    return {
        # Animal events with no box score 0; if too many, no threshold above 0 meets the target.
        "reachable": bool((val_animal > 0).mean() >= target),
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
    ap.add_argument(
        "--ours",
        nargs="+",
        default=[
            "ours=outputs/phase2/classifier/predictions.csv",
            "ours-nols=outputs/phase2/classifier_nols/predictions.csv",
        ],
        help="name=path of our classifiers' predictions; missing files are skipped",
    )
    ap.add_argument("--baselines", default="outputs/phase2/baselines")
    ap.add_argument("--reports", default="reports/phase2")
    args = ap.parse_args()

    cfg = load_config(args.config)
    target = cfg["evaluation"]["target_recall"]
    models, covers = {}, {}
    for spec in args.ours:
        name, path = spec.split("=", 1)
        if Path(path).exists():
            models[name] = pd.read_csv(path)
            covers[name] = set(models[name].filter(regex="^p_").columns.str[2:])
        else:
            print(f"(no {path}, skipping {name})")
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
    crops = next(iter(models.values()))
    events = load_events(Path(args.manifest), Path(args.splits))
    print(f"{len(shared):,} crops in {crops['seq_key'].nunique():,} events; models: {', '.join(models)}")

    methods = {"detector": crops["conf"]}
    for name, df in models.items():
        p_animal = 1.0 - df[f"p_{BACKGROUND}"]
        methods[name] = p_animal
        methods[f"{name} x detector"] = p_animal * crops["conf"]

    targets = sorted({target, 0.95, 0.90}, reverse=True)
    rows, by_group = [], {}
    for name, score in methods.items():
        for agg, k in AGGREGATIONS.items():
            ev = events.assign(score=event_scores(crops, score, events, k).to_numpy())
            for t in targets:
                rows.append({"target": t, "method": name, "events_by": agg, **evaluate(ev, t)})
            if " x " not in name:
                test = ev[ev["split"] == "test"]
                t_low = rows[-1]["threshold"]  # loosest target: the strict ones may be out of reach
                by_group[f"{name}, {agg}"] = recall_by_group(test, t_low).set_index("group")["recall"]
    table = pd.DataFrame(rows)
    groups_recall = pd.DataFrame(by_group)
    cell = table.assign(
        cell=[
            f"{r.empty_removed:.1%} ({r.test_recall:.1%})" if r.reachable else "out of reach"
            for r in table.itertuples()
        ],
        method=table["method"] + ", " + table["events_by"],
    )
    summary = cell.pivot(index="method", columns="target", values="cell")[targets]
    summary = summary.reindex(dict.fromkeys(cell["method"]))
    summary.columns = [f"{t:.0%} target" for t in targets]
    summary.index.name = "method, events scored by"

    # Per-class F1 on test crops the classifier was meant to handle, over classes every model can name.
    common = sorted(set.intersection(*covers.values()))
    test_ok = crops[(crops["split"] == "test") & crops["train_ok"] & crops["label"].isin(common)]
    names = common + ["(other)"]
    idx = {g: k for k, g in enumerate(common)}
    f1 = {}
    for name, df in models.items():
        sub = df.loc[test_ok.index, [f"p_{g}" for g in common]]
        pred = sub.to_numpy().argmax(1)
        # The top choice may be a class not every model has: count that as "other".
        rest = df.loc[test_ok.index].filter(regex="^p_").drop(columns=sub.columns)
        if not rest.empty:
            pred = np.where(rest.max(axis=1).to_numpy() > sub.max(axis=1).to_numpy(), len(common), pred)
        rep = per_class_report(test_ok["label"].map(idx), np.asarray(pred, dtype=int), names)
        f1[name] = rep.set_index("class")["f1"].drop(["(other)", "macro avg"])
    f1 = pd.DataFrame(f1)

    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    table.to_csv(reports / "event_eval.csv", index=False)
    n_test = events[events["split"] == "test"]
    n_animal, n_empty = (n_test["gt"] == "animal").sum(), (n_test["gt"] == "empty").sum()
    n_cams = n_test["location"].nunique()
    n_nobox = int(((n_test["gt"] == "animal") & ~n_test["seq_key"].isin(set(crops["seq_key"]))).sum())
    lines = [
        "# Phase 2: empty-event filtering with a crop classifier",
        "",
        f"Test: {len(n_test):,} events ({n_animal:,} animal, {n_empty:,} empty) from {n_cams} cameras not used in training.",
        "Each threshold is the highest that keeps the target share of animal events on val, applied unchanged to test.",
        "",
        (
            f"{n_nobox} of {n_animal} animal test events have no detector box at or above the crop floor, so no method can keep more "
            f"than {1 - n_nobox / n_animal:.1%} of them. Targets above what val allows are marked out of reach."
        ),
        "",
        "Cells: share of empty test events removed (test recall of animal events in brackets).",
        "Thresholds, workload and missed counts are in event_eval.csv.",
        "",
        markdown(summary, str),
        "",
        f"## Test recall by species group at the {targets[-1]:.0%} threshold",
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
