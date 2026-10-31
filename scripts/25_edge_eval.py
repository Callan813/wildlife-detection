"""Phase 6, step 2b: how much accuracy does each ONNX variant lose, on the original test sets?

    python scripts/25_edge_eval.py --models detector --variants int8
    python scripts/25_edge_eval.py --models reid species --variants fp32 int8

The reference is always the PyTorch run from the earlier phases, so a variant is compared
on exactly the data and metric it was judged on before:

- detector: the 5,726 Phase 1 frames (1,200 trigger events). Event AUC, recall at the 0.01
  floor, and the 90%-recall operating point. The threshold is picked on the PyTorch scores
  and applied unchanged to the variant too, as a deployed camera would do: a threshold
  tuned on the full model is only safe if the quantised scores sit in the same place.
- reid: Leopard ID 2022 test crops, same retrieval rules as Phase 3 (top-k per leopard,
  mAP), plus the cosine similarity between each variant embedding and the PyTorch one.
- species: Phase 2 test-camera crops. The probe head is refit on the cached PyTorch
  features of train cameras (C as in Phase 2) and applied to the variant's features, so
  any change in macro F1 comes from the quantised backbone.

Outputs go to outputs/phase6/eval/ (per-variant detections / features, resumable for the
detector) and a summary table to reports/phase6/accuracy.md.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ghats.edge import decode, letterbox, to_detections

OUT = Path("outputs/phase6/eval")


def session(path: Path, threads: int):
    import onnxruntime as ort

    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])


# ---------------------------------------------------------------- detector


def load_frame(row):
    image_id, path = row
    with Image.open(path) as im:
        x, rev = letterbox(im)
        return image_id, x, rev, im.size


def run_detector(model: Path, out: Path, threads: int, batch: int = 8) -> None:
    manifest = pd.read_csv("data/swg/manifest.csv")
    done = set()
    if out.exists():
        done = {json.loads(line)["image_id"] for line in out.read_text(encoding="utf-8").splitlines() if line}
    rows = [(i, p) for i, p in zip(manifest["image_id"], manifest["local_path"]) if i not in done]
    print(f"{model.name}: {len(rows):,} frames to run ({len(done):,} done)")
    sess = session(model, threads)
    start = time.perf_counter()
    with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(2) as pool:
        for b in range(0, len(rows), batch):
            items = list(pool.map(load_frame, rows[b : b + batch]))
            t = time.perf_counter()
            scores, boxes = sess.run(None, {"input": np.stack([it[1] for it in items])})
            ms = (time.perf_counter() - t) * 1000 / len(items)
            for (image_id, _, rev, (w, h)), s, bx in zip(items, scores, boxes):
                dets = to_detections(decode(s, bx), rev, w, h)
                rec = {
                    "image_id": image_id,
                    "model": model.stem,
                    "ms": round(ms, 2),
                    "detections": [d.__dict__ for d in dets],
                }
                f.write(json.dumps(rec) + "\n")
            if (b // batch) % 50 == 0:
                rate = (b + len(items)) / (time.perf_counter() - start)
                print(f"  {b + len(items):,}/{len(rows):,}  {rate:.1f} img/s", flush=True)


def event_scores(path: Path) -> pd.Series:
    manifest = pd.read_csv("data/swg/manifest.csv")
    score = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        score[r["image_id"]] = max((d["conf"] for d in r["detections"] if d["category"] == "animal"), default=0.0)
    m = manifest.assign(score=manifest["image_id"].map(score))
    if m["score"].isna().any():
        raise SystemExit(f"{path} is missing {m['score'].isna().sum()} frames")
    return m.groupby("seq_key").agg(score=("score", "max"), animal=("seq_gt", lambda g: g.iloc[0] == "animal"))


def detector_metrics(ev: pd.DataFrame, ref_thr90: float) -> dict:
    from sklearn.metrics import roc_auc_score

    from ghats.metrics import threshold_for_recall

    a, e = ev.loc[ev["animal"], "score"], ev.loc[~ev["animal"], "score"]
    own90 = threshold_for_recall(a, 0.9)
    return {
        "event AUC": roc_auc_score(ev["animal"], ev["score"]),
        "recall at 0.01": (a >= 0.01).mean(),
        "empties removed at 0.01": (e < 0.01).mean(),
        "own 90% threshold": own90,
        "empties removed at own 90%": (e < own90).mean(),
        "recall at reference 90% threshold": (a >= ref_thr90).mean(),
        "empties removed at reference threshold": (e < ref_thr90).mean(),
    }


def detector(variants: list[str], threads: int) -> pd.DataFrame:
    from ghats.metrics import threshold_for_recall

    ref = event_scores(Path("outputs/phase1/detections.jsonl"))
    thr = threshold_for_recall(ref.loc[ref["animal"], "score"], 0.9)
    res = {"pytorch (GPU)": detector_metrics(ref, thr)}
    for v in variants:
        path = OUT / f"detections_{v}.jsonl"
        run_detector(Path(f"outputs/phase6/onnx/detector_{v}.onnx"), path, threads)
        ev = event_scores(path)
        res[v] = detector_metrics(ev, thr) | {
            "events whose score moved > 0.05": (ev["score"] - ref["score"]).abs().gt(0.05).mean()
        }
    return pd.DataFrame(res)


# ---------------------------------------------------------------- re-ID


def reid(variants: list[str], threads: int) -> pd.DataFrame:
    import torch  # noqa: F401  (torchvision transforms, for exactly the Phase 3 preprocessing)

    from ghats.config import load_config
    from ghats.reid import l2_normalise, retrieval_metrics
    from ghats.reid_model import Crops, transforms

    cfg = load_config("configs/phase3.yaml")
    crops = pd.read_csv(Path(cfg["crops"]["out"]) / "crops.csv").query("split == 'test'").reset_index(drop=True)
    paths = [Path(cfg["crops"]["out"]) / "images" / f"{c}.jpg" for c in crops["crop_id"]]
    tf = transforms(tuple(cfg["train"]["size"]), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 0, train=False)
    ds = Crops(paths, [0] * len(paths), tf)
    ref = np.load("outputs/phase3/reid_lp102_s1/test_embeddings.npz")["x"]
    ks = cfg["reid"]["ks"]

    def score(emb):
        m = retrieval_metrics(emb, crops["name"], crops["encounter"], crops["viewpoint"], ks)
        return {f"top-{k} per leopard": m[f"top{k}_per_id"] for k in ks} | {"mAP": m["mAP"]}

    res = {"pytorch (GPU)": score(ref)}
    for v in variants:
        cache = OUT / f"reid_{v}.npy"
        if not cache.exists():
            sess = session(Path(f"outputs/phase6/onnx/reid_{v}.onnx"), threads)
            embs = []
            for b in range(0, len(ds), 16):
                x = np.stack([ds[i][0].numpy() for i in range(b, min(b + 16, len(ds)))])
                embs.append(sess.run(None, {"input": x})[0])
            np.save(cache, l2_normalise(np.concatenate(embs)))
        emb = np.load(cache)
        res[v] = score(emb) | {"min cosine to pytorch": float((emb * ref).sum(1).min())}
    return pd.DataFrame(res)


# ---------------------------------------------------------------- species


def species(variants: list[str], threads: int, c: float = 0.01, batch: int = 4) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from ghats.baselines import TightCrops, speciesnet_preprocess
    from ghats.classify import class_weights, per_class_report

    z = np.load("outputs/phase2/speciesnet_probe/features.npz", allow_pickle=True)
    feats = pd.DataFrame(z["x"], index=z["crop_id"])
    crops = pd.read_csv("data/crops/crops.csv", dtype={"image_id": str})
    crops = crops[crops["crop_id"].isin(feats.index)]
    groups = sorted(crops["label"].unique())
    idx = {g: k for k, g in enumerate(groups)}
    tr = crops[(crops["split"] == "train") & crops["train_ok"]]
    te = crops[(crops["split"] == "test") & crops["train_ok"]].reset_index(drop=True)

    scaler = StandardScaler().fit(feats.loc[tr["crop_id"]])
    y_tr = tr["label"].map(idx).to_numpy()
    sw = class_weights(np.bincount(y_tr, minlength=len(groups)))[y_tr]
    clf = LogisticRegression(C=c, max_iter=3000).fit(scaler.transform(feats.loc[tr["crop_id"]]), y_tr, sample_weight=sw)
    y_te = te["label"].map(idx).to_numpy()

    def score(x):
        pred = clf.predict(scaler.transform(x))
        return {"macro F1": per_class_report(y_te, pred, groups)["f1"].iloc[-1], "accuracy": (pred == y_te).mean()}

    ref = feats.loc[te["crop_id"]].to_numpy()
    res = {"pytorch (GPU)": score(ref)}
    paths = pd.read_csv("data/swg/manifest_phase2.csv", dtype={"image_id": str}).set_index("image_id")["local_path"]
    ds = TightCrops(
        te["image_id"].map(paths).tolist(), [ast.literal_eval(s) for s in te["bbox"]], speciesnet_preprocess
    )
    for v in variants:
        cache = OUT / f"species_{v}.npy"
        if not cache.exists():
            # 480x480 inputs are heavy: small batches, and a part file every 256 crops so a
            # killed run picks up where it stopped
            parts = OUT / f"species_{v}_parts"
            parts.mkdir(exist_ok=True)
            sess = session(Path(f"outputs/phase6/onnx/species_{v}.onnx"), threads)
            for start in range(0, len(ds), 256):
                part = parts / f"{start:05d}.npy"
                if part.exists():
                    continue
                stop = min(start + 256, len(ds))
                out = []
                for b in range(start, stop, batch):
                    x = np.stack([ds[i].numpy() for i in range(b, min(b + batch, stop))])
                    out.append(sess.run(None, {"input": x})[0])
                np.save(part, np.concatenate(out))
                print(f"  species {v}: {stop:,}/{len(ds):,}", flush=True)
            np.save(cache, np.concatenate([np.load(p) for p in sorted(parts.glob("*.npy"))]))
        x = np.load(cache)
        cos = (x * ref).sum(1) / np.linalg.norm(x, axis=1) / np.linalg.norm(ref, axis=1)
        res[v] = score(x) | {
            "min cosine to pytorch": float(cos.min()),
            "prediction agrees with pytorch": (
                clf.predict(scaler.transform(x)) == clf.predict(scaler.transform(ref))
            ).mean(),
        }
    return pd.DataFrame(res)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["detector", "reid", "species"])
    ap.add_argument("--variants", nargs="+", default=["int8"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--reports", default="reports/phase6")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    run = {"detector": detector, "reid": reid, "species": species}
    for name in args.models:
        table = run[name](args.variants, args.threads)
        table.to_csv(reports / f"accuracy_{name}.csv")
        print(f"\n## {name}\n\n{table.to_markdown(floatfmt='.3f')}\n", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
