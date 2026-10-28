"""Phase 1, step 3: how well does MegaDetector filter empty frames?

    python scripts/03_evaluate.py --manifest data/swg/manifest.csv \
        --detections outputs/phase1/detections.jsonl --out reports/phase1

Reports metrics at two levels:
  * image level: every frame on its own (inflated misses, see note below)
  * event level: one camera trigger = one unit (the number that matters)

Note: LILA labels are often per sequence, so a frame where the animal has
already walked out of view may still carry an animal label. Image-level
"misses" therefore include label noise; event-level metrics do not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ghats.config import load_config
from ghats.metrics import pick_operating_point, recall_by_group, threshold_sweep


def load_detections(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            animal = [d["conf"] for d in rec["detections"] if d["category"] == "animal"]
            rows.append(
                {
                    "image_id": str(rec["image_id"]),
                    "score": max(animal) if animal else 0.0,
                    "n_animal_boxes": len(animal),
                    "ms": rec.get("ms"),
                    "model": rec.get("model"),
                }
            )
    return pd.DataFrame(rows).drop_duplicates("image_id", keep="last")


def evaluate(manifest: pd.DataFrame, dets: pd.DataFrame, thresholds: list[float], target_recall: float) -> dict:
    df = manifest.merge(dets, on="image_id", how="inner")
    if df.empty:
        raise SystemExit("No overlap between manifest and detections. Did step 2 run?")

    image_sweep = threshold_sweep(df["score"], df["gt"] == "animal", thresholds)

    events = (
        df.groupby("seq_key")
        .agg(
            gt=("seq_gt", "first"),
            group=("seq_group", "first"),
            score=("score", "max"),
            n_frames=("image_id", "size"),
            location=("location", "first"),
        )
        .reset_index()
    )
    event_sweep = threshold_sweep(events["score"], events["gt"] == "animal", thresholds)
    op = pick_operating_point(event_sweep, target_recall)
    groups = recall_by_group(events, float(op["threshold"]))

    ms = df["ms"].dropna().to_numpy(dtype=float)
    latency = {
        "mean_ms": float(ms.mean()) if ms.size else float("nan"),
        "median_ms": float(np.median(ms)) if ms.size else float("nan"),
        "p95_ms": float(np.percentile(ms, 95)) if ms.size else float("nan"),
    }
    return {
        "frames": df,
        "events": events,
        "image_sweep": image_sweep,
        "event_sweep": event_sweep,
        "operating_point": op,
        "groups": groups,
        "latency": latency,
        "model": df["model"].dropna().iloc[0] if df["model"].notna().any() else "unknown",
    }


def plot_tradeoff(image_sweep: pd.DataFrame, event_sweep: pd.DataFrame, op: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)
    for sweep, label, style in [(event_sweep, "Per event", "-o"), (image_sweep, "Per frame", "--s")]:
        ax.plot(sweep["review_reduction"] * 100, sweep["recall"] * 100, style, ms=3.5, lw=1.6, label=label)
    ax.scatter(
        [op["review_reduction"] * 100],
        [op["recall"] * 100],
        s=90,
        facecolors="none",
        edgecolors="black",
        linewidths=1.4,
        zorder=5,
    )
    ax.annotate(
        f"threshold {op['threshold']:.2f}",
        (op["review_reduction"] * 100, op["recall"] * 100),
        textcoords="offset points",
        xytext=(8, -14),
        fontsize=8,
    )
    ax.set_xlabel("Images a human no longer needs to review (%)")
    ax.set_ylabel("Animal units kept (recall, %)")
    ax.set_title("Empty-frame filtering trade-off")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fmt_pct(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x * 100:.1f}%"


def write_summary(r: dict, path: Path, target_recall: float) -> None:
    op, frames, events = r["operating_point"], r["frames"], r["events"]
    n_ev_animal = int((events["gt"] == "animal").sum())
    lines = [
        "# Phase 1 results: empty-frame filtering",
        "",
        f"Model: `{r['model']}`",
        (
            f"Data: {len(frames):,} frames in {len(events):,} trigger events "
            f"({n_ev_animal:,} animal, {len(events) - n_ev_animal:,} empty) "
            f"from {frames['location'].nunique()} camera locations."
        ),
        "",
        f"## Operating point (target event recall {target_recall:.0%})",
        "",
    ]
    if op["recall"] < target_recall:
        lines += [
            (
                f"The target is out of reach: even at the lowest threshold, {int(op['fn'])} animal events "
                "have no detector box at all. The row below is the lowest threshold."
            ),
            "",
        ]
    lines += [
        "| Metric | Value |",
        "|---|---|",
        f"| Confidence threshold | {op['threshold']:.2f} |",
        f"| Animal events kept (recall) | {fmt_pct(op['recall'])} |",
        f"| Precision of kept events | {fmt_pct(op['precision'])} |",
        f"| Empty events removed | {fmt_pct(op['empty_removed'])} |",
        f"| Review workload removed | {fmt_pct(op['review_reduction'])} |",
        f"| Animal events missed | {int(op['fn'])} of {n_ev_animal} |",
        "",
        "## Recall by species group at that threshold (worst first)",
        "",
        "| Group | Events | Recall |",
        "|---|---|---|",
    ]
    lines += [f"| {g.group} | {g.n} | {fmt_pct(g.recall)} |" for g in r["groups"].itertuples()]
    lat = r["latency"]
    lines += [
        "",
        "## Latency (this machine, per image, includes image decode; batched runs report batch time / batch size)",
        "",
        f"Mean {lat['mean_ms']:.0f} ms, median {lat['median_ms']:.0f} ms, p95 {lat['p95_ms']:.0f} ms.",
        "",
        "## Files",
        "",
        "- `event_sweep.csv`, `image_sweep.csv`: metrics at every threshold",
        "- `recall_by_group.csv`: per-group recall at the operating point",
        "- `tradeoff.png`: recall vs. review workload removed",
        "",
        (
            "Frame-level numbers include label noise (sequence-level labels on frames where the animal "
            "has left the view), so the event-level row is the headline result."
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--manifest", default="data/swg/manifest.csv")
    ap.add_argument("--detections", default="outputs/phase1/detections.jsonl")
    ap.add_argument("--out", default="reports/phase1")
    args = ap.parse_args()

    ecfg = load_config(args.config)["evaluation"]
    manifest = pd.read_csv(args.manifest, dtype={"image_id": str})
    dets = load_detections(Path(args.detections))
    missing = len(set(manifest["image_id"]) - set(dets["image_id"]))
    if missing:
        print(f"Note: {missing:,} manifest images have no detections yet and are skipped.")

    r = evaluate(manifest, dets, ecfg["thresholds"], ecfg["target_recall"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    r["image_sweep"].to_csv(out / "image_sweep.csv", index=False)
    r["event_sweep"].to_csv(out / "event_sweep.csv", index=False)
    r["groups"].to_csv(out / "recall_by_group.csv", index=False)
    r["frames"][["image_id", "local_path", "seq_key", "gt", "seq_gt", "seq_group", "score"]].to_csv(
        out / "frame_scores.csv", index=False
    )
    plot_tradeoff(r["image_sweep"], r["event_sweep"], r["operating_point"], out / "tradeoff.png")
    write_summary(r, out / "summary.md", ecfg["target_recall"])
    print((out / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
