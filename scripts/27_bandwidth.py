"""Phase 6, step 4: how much does filtering on the camera save in data sent?

    python scripts/27_bandwidth.py

A remote camera trap pays for every byte it sends (cellular or satellite) or has to be
visited to swap SD cards. This compares what the camera sends for the Phase 2 test events
(566 triggers from 106 cameras the models never saw) under five policies:

  every frame                          no filtering, the baseline
  frames of events the detector keeps  MegaDetector on the camera, Phase 1 rule
  frames of events the cascade keeps   detector + species head (the live pipeline)
  one crop per kept event + metadata   what scripts/26_live.py sends
  metadata only                        a line of JSON per kept event (species, time, P)

Thresholds are the Phase 2 operating points for 90% of animal events, picked on val
cameras (reports/phase2/event_eval.csv), so animal recall is the same honest test number.
Frame sizes are the real JPEG files; crop sizes are the saved Phase 2 crops, which use the
same settings as the live pipeline (padded box, longest side 384 px, quality 90).

Writes reports/phase6/bandwidth.{md,csv}.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from ghats.crops import BACKGROUND

META_BYTES = 300  # one events.jsonl line from 26_live.py is ~250-330 bytes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/swg/manifest_phase2.csv")
    ap.add_argument("--splits", default="data/crops/splits.csv")
    ap.add_argument("--predictions", default="outputs/phase2/speciesnet_probe/predictions.csv")
    ap.add_argument("--crops", default="data/crops/images")
    ap.add_argument("--event-eval", default="reports/phase2/event_eval.csv")
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--reports", default="reports/phase6")
    args = ap.parse_args()

    split = pd.read_csv(args.splits).set_index("location")["split"]
    m = pd.read_csv(args.manifest, dtype={"image_id": str})
    m = m[m["location"].map(split) == "test"].copy()
    m["bytes"] = [os.path.getsize(p) for p in m["local_path"]]
    events = m.groupby("seq_key").agg(
        frames=("image_id", "size"), frame_bytes=("bytes", "sum"), animal=("seq_gt", lambda g: g.iloc[0] == "animal")
    )

    p = pd.read_csv(args.predictions)
    p = p[p["split"] == "test"]
    p["p_animal"] = 1.0 - p[f"p_{BACKGROUND}"]
    best_det = p.groupby("seq_key")["conf"].max()
    best = p.loc[p.groupby("seq_key")["p_animal"].idxmax()].set_index("seq_key")
    best["crop_bytes"] = [os.path.getsize(Path(args.crops) / f"{c}.jpg") for c in best["crop_id"]]
    events["det"] = events.index.map(best_det).fillna(0.0)
    events["p_animal"] = events.index.map(best["p_animal"]).fillna(0.0)
    events["crop_bytes"] = events.index.map(best["crop_bytes"]).fillna(0)

    ev = pd.read_csv(args.event_eval)
    pick = ev[(ev["target"] == args.target) & (ev["events_by"] == "best box")].set_index("method")["threshold"]
    t_det, t_casc = float(pick["detector"]), float(pick["probe"])
    kept_det = events["det"] >= t_det
    kept_casc = events["p_animal"] >= t_casc

    def row(name: str, kept: pd.Series, sent: float) -> dict:
        return {
            "policy": name,
            "events sent": int(kept.sum()),
            "animal events kept": kept[events["animal"]].mean(),
            "empty events sent": kept[~events["animal"]].mean(),
            "MB sent": sent / 1e6,
            "share of every frame": sent / events["frame_bytes"].sum(),
            "KB per animal event kept": sent / 1e3 / max(int((kept & events["animal"]).sum()), 1),
        }

    everything = pd.Series(True, index=events.index)
    rows = [
        row("every frame", everything, events["frame_bytes"].sum()),
        row(
            f"frames of events the detector keeps (conf >= {t_det:.3f})",
            kept_det,
            events.loc[kept_det, "frame_bytes"].sum(),
        ),
        row(
            f"frames of events the cascade keeps (P(animal) >= {t_casc:.3f})",
            kept_casc,
            events.loc[kept_casc, "frame_bytes"].sum(),
        ),
        row(
            "one crop per kept event + metadata",
            kept_casc,
            events.loc[kept_casc, "crop_bytes"].sum() + META_BYTES * kept_casc.sum(),
        ),
        row("metadata only", kept_casc, META_BYTES * kept_casc.sum()),
    ]
    table = pd.DataFrame(rows)

    # The test set is 74% animal events by design (sampled per species). Many real deployments
    # see mostly empty triggers (wind, sun, moving shadows), so project MB per 1,000 triggers from
    # the per-event rates measured here, at other shares of empty triggers.
    def per_trigger(mask: pd.Series, sent: pd.Series, animal: bool) -> float:
        sel = events["animal"] == animal
        return float((sent * mask)[sel].mean())

    frames, crops = events["frame_bytes"], events["crop_bytes"] + META_BYTES
    policies = {
        "every frame": (everything, frames),
        "frames, detector filter": (kept_det, frames),
        "frames, cascade filter": (kept_casc, frames),
        "crop + metadata, cascade": (kept_casc, crops),
    }
    scen = pd.DataFrame(
        {
            f"{e:.0%} empty": {
                name: ((1 - e) * per_trigger(k, s, True) + e * per_trigger(k, s, False)) * 1000 / 1e6
                for name, (k, s) in policies.items()
            }
            for e in (0.26, 0.5, 0.8, 0.95)
        }
    )

    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    table.to_csv(reports / "bandwidth.csv", index=False)
    n_animal, n_empty = int(events["animal"].sum()), int((~events["animal"]).sum())
    mean_frame = m["bytes"].mean() / 1e6
    lines = [
        "# Phase 6: data sent by a camera, with and without on-camera filtering",
        "",
        (
            f"Phase 2 test set: {len(events)} trigger events ({n_animal} animal, {n_empty} empty), "
            f"{len(m):,} frames (mean {mean_frame:.2f} MB), from {m['location'].nunique()} cameras not used in training."
        ),
        f"Thresholds keep {args.target:.0%} of animal events on val cameras (reports/phase2/event_eval.csv).",
        "",
        "| Policy | Events sent | Animal events kept | Empty events sent | MB sent | Share of every frame | KB per animal event |",
        "|---|---|---|---|---|---|---|",
        *(
            f"| {r['policy']} | {r['events sent']} | {r['animal events kept']:.1%} | {r['empty events sent']:.1%} | "
            f"{r['MB sent']:,.1f} | {r['share of every frame']:.2%} | {r['KB per animal event kept']:,.1f} |"
            for r in rows
        ),
        "",
        "## MB per 1,000 triggers, by share of empty triggers",
        "",
        "Projected from the per-event rates above (26% empty is this test set).",
        "",
        "| Policy | " + " | ".join(scen.columns) + " |",
        "|---|" + "---|" * len(scen.columns),
        *(f"| {name} | " + " | ".join(f"{v:,.0f}" for v in r) + " |" for name, r in scen.iterrows()),
        "",
        "Events with no box at all are never sent; that's most of the missed animal events (see Phase 2).",
        f"Metadata is counted as {META_BYTES} bytes per event (an events.jsonl line from scripts/26_live.py).",
    ]
    (reports / "bandwidth.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
