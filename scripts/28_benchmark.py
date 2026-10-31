"""Phase 6, step 3: how fast, and how big, is each model on this machine's CPU?

    python scripts/28_benchmark.py                       # all models, 1/2/4 threads
    python scripts/28_benchmark.py --threads 4 --runs 30

Needs only numpy and onnxruntime (psutil, if installed, adds memory), so the same command
runs on a Raspberry Pi: copy outputs/phase6/onnx/ and this script, pip install numpy
onnxruntime, run it. Inputs are random tensors of each model's real shape; speed doesn't
depend on the pixel values for these networks.

For each model and thread count: load time, median and p95 latency per call (batch 1, after
warm-up), and resident memory after loading and running. Results are tagged with the machine,
because laptop numbers are not Pi numbers. Writes reports/phase6/benchmark_<machine>.{md,csv}.
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

SHAPES = {"detector": (1, 3, 640, 640), "species": (1, 480, 480, 3), "reid": (1, 3, 224, 448)}


def rss_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process().memory_info().rss / 2**20


def machine() -> str:
    cpu = platform.processor() or platform.machine()
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith(("model name", "model")):
                    cpu = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    return f"{cpu}, {os.cpu_count()} logical CPUs, {platform.system()} {platform.machine()}"


def bench(path: Path, shape: tuple[int, ...], threads: int, runs: int) -> dict:
    import onnxruntime as ort

    gc.collect()
    before = rss_mb()
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    t = time.perf_counter()
    sess = ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])
    load = time.perf_counter() - t
    x = np.random.default_rng(0).random(shape, dtype=np.float32)
    name = sess.get_inputs()[0].name
    for _ in range(3):
        sess.run(None, {name: x})
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        sess.run(None, {name: x})
        times.append((time.perf_counter() - t) * 1000)
    after = rss_mb()
    del sess
    return {
        "load s": load,
        "median ms": float(np.median(times)),
        "p95 ms": float(np.percentile(times, 95)),
        "per second": 1000 / float(np.median(times)),
        "memory MB": (after - before) if before is not None and after is not None else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="outputs/phase6/onnx")
    ap.add_argument(
        "--models", nargs="+", default=["detector_fp32", "detector_int8", "species_fp32", "reid_fp32", "reid_int8"]
    )
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--tag", help="name for the report file (default: this machine's name)")
    ap.add_argument("--reports", default="reports/phase6")
    args = ap.parse_args()

    import onnxruntime as ort

    host = machine()
    print(f"{host}; onnxruntime {ort.__version__}")
    rows = []
    for m in args.models:
        path = Path(args.dir) / f"{m}.onnx"
        if not path.exists():
            print(f"(no {path}, skipping)")
            continue
        for th in args.threads:
            r = bench(path, SHAPES[m.split("_")[0]], th, args.runs)
            rows.append({"model": m, "MB on disk": path.stat().st_size / 2**20, "threads": th, **r})
            mem = f"{r['memory MB']:.0f} MB" if r["memory MB"] is not None else "n/a"
            print(
                f"{m:15s} {th} threads: {r['median ms']:7.1f} ms median, {r['p95 ms']:7.1f} p95, "
                f"load {r['load s']:.1f} s, memory {mem}",
                flush=True,
            )

    tag = args.tag or platform.node().lower().replace(" ", "-")
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    with open(reports / f"benchmark_{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["model"])
        w.writeheader()
        w.writerows(rows)
    lines = [
        f"# Phase 6: model speed on {tag}",
        "",
        f"{host}; onnxruntime {ort.__version__}, CPU execution provider, batch 1, {args.runs} runs after 3 warm-up.",
        "Not a Raspberry Pi: rerun this script on the device for its numbers.",
        "",
        "| Model | MB on disk | Threads | Median ms | p95 ms | Calls per second | Load s | Memory MB |",
        "|---|---|---|---|---|---|---|---|",
        *(
            f"| {r['model']} | {r['MB on disk']:.1f} | {r['threads']} | {r['median ms']:.0f} | {r['p95 ms']:.0f} | "
            f"{r['per second']:.2f} | {r['load s']:.1f} | "
            + (f"{r['memory MB']:.0f}" if r["memory MB"] is not None else "n/a")
            + " |"
            for r in rows
        ),
    ]
    (reports / f"benchmark_{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
