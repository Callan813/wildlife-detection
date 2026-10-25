"""End-to-end run of steps 1, 3 and 4 on synthetic data (no network, no torch)."""

import json
import random
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CATS = [
    {"id": 0, "name": "empty"},
    {"id": 1, "name": "sambar"},
    {"id": 2, "name": "unidentified_murid"},
    {"id": 3, "name": "blurred"},
    {"id": 4, "name": "eurasian_wild_pig"},
]


def run(script: str, *args: str) -> str:
    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args], capture_output=True, text=True, check=True
    )
    return res.stdout


def test_phase1_pipeline(tmp_path):
    rng = random.Random(0)
    images, anns, k = [], [], 0
    for loc in range(12):
        for s in range(8):
            cat = rng.choice([0, 0, 1, 2, 3, 4])
            for f in range(3):
                iid = f"img{k}"
                k += 1
                images.append(
                    {
                        "id": iid,
                        "file_name": f"public/loc_{loc}/{iid}.jpg",
                        "location": f"loc_{loc}",
                        "datetime": f"2019-06-{1 + s:02d} 10:00:{f * 2:02d}",
                        "seq_id": f"s{loc}_{s}",
                    }
                )
                anns.append({"id": f"a{iid}", "image_id": iid, "category_id": cat})
    (tmp_path / "meta.json").write_text(json.dumps({"images": images, "annotations": anns, "categories": CATS}))

    cfg = yaml.safe_load((ROOT / "configs" / "phase1.yaml").read_text())
    cfg["sampling"].update(n_animal_sequences=20, n_empty_sequences=20, max_sequences_per_location=4)
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))

    data = tmp_path / "data"
    run(
        "01_fetch_subset.py",
        "--config",
        str(tmp_path / "cfg.yaml"),
        "--out",
        str(data),
        "--metadata",
        str(tmp_path / "meta.json"),
        "--no-images",
    )
    manifest = pd.read_csv(data / "manifest.csv", dtype={"image_id": str})
    assert set(manifest["gt"]) <= {"animal", "empty"}, "excluded frames must not be sampled"
    assert manifest.groupby("seq_key").size().eq(3).all(), "whole sequences must be sampled"
    assert manifest.groupby("location")["seq_key"].nunique().max() <= 4

    # A perfect fake detector: animals score 0.9, empties 0.0
    dets = tmp_path / "dets.jsonl"
    with open(dets, "w") as f:
        for r in manifest.itertuples():
            Path(r.local_path).parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 48)).save(r.local_path)
            boxes = [{"category": "animal", "conf": 0.9, "bbox": [0.1, 0.1, 0.5, 0.5]}] if r.gt == "animal" else []
            f.write(json.dumps({"image_id": r.image_id, "model": "fake", "ms": 10.0, "detections": boxes}) + "\n")

    reports = tmp_path / "reports"
    out = run(
        "03_evaluate.py",
        "--config",
        str(tmp_path / "cfg.yaml"),
        "--manifest",
        str(data / "manifest.csv"),
        "--detections",
        str(dets),
        "--out",
        str(reports),
    )
    assert "| Animal events kept (recall) | 100.0% |" in out
    assert "| Empty events removed | 100.0% |" in out
    for name in ["event_sweep.csv", "image_sweep.csv", "recall_by_group.csv", "tradeoff.png", "summary.md"]:
        assert (reports / name).exists()

    run(
        "04_gallery.py",
        "--scores",
        str(reports / "frame_scores.csv"),
        "--detections",
        str(dets),
        "--out",
        str(reports),
        "--threshold",
        "0.95",
    )
    assert (reports / "missed_animals.jpg").exists()
