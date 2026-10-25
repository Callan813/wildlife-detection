"""Phase 1, step 1: download LILA metadata and a balanced, sequence-level sample.

    python scripts/01_fetch_subset.py --config configs/phase1.yaml --out data/swg

Writes data/swg/manifest.csv and data/swg/images/... The download resumes:
files already on disk are skipped.
"""

from __future__ import annotations

import argparse
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from ghats.coco_ct import GT_ANIMAL, GT_EMPTY, GT_EXCLUDED, build_image_table, load_coco_ct
from ghats.config import load_config
from ghats.events import assign_events


def download(url: str, dest: Path, session: requests.Session, retries: int = 3) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for _ in range(retries):
        try:
            with session.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    f.writelines(r.iter_content(1 << 16))
            tmp.replace(dest)
            return True
        except requests.RequestException:
            continue
    tmp.unlink(missing_ok=True)
    return False


def fetch_metadata(url: str, out_dir: Path) -> Path:
    dest = out_dir / Path(url).name
    if dest.exists():
        print(f"Metadata already downloaded: {dest}")
        return dest
    print(f"Downloading metadata (this can be a few hundred MB): {url}")
    with requests.Session() as s, s.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
        tmp.replace(dest)
    return dest


def sequence_table(images: pd.DataFrame) -> pd.DataFrame:
    """One row per sequence. Sequences with any excluded frame are dropped."""
    seqs = images.groupby("seq_key").agg(
        n_frames=("image_id", "size"),
        location=("location", "first"),
        has_excluded=("gt", lambda s: (s == GT_EXCLUDED).any()),
        has_animal=("gt", lambda s: (s == GT_ANIMAL).any()),
        group=("group", lambda s: s.dropna().iloc[0] if s.notna().any() else None),
    )
    seqs = seqs[~seqs["has_excluded"]].copy()
    seqs["gt"] = seqs["has_animal"].map({True: GT_ANIMAL, False: GT_EMPTY})
    return seqs.reset_index()


def capped_sample(
    df: pd.DataFrame,
    n: int,
    rng: random.Random,
    per_location: int,
    per_group: int | None = None,
    by_loc: dict[str, int] | None = None,
) -> list[str]:
    """Random sample of sequence keys with per-location (and per-group) caps.

    Pass the same ``by_loc`` dict to several calls so the location cap covers
    the animal and empty samples together.
    """
    order = df.sample(frac=1.0, random_state=rng.randint(0, 2**31 - 1))
    if per_group is not None:
        # Round-robin across groups: otherwise common groups fill n before rare
        # ones (small cats, pangolin) are ever reached.
        order = order.assign(_rank=order.groupby("group").cumcount()).sort_values("_rank", kind="stable")
    by_loc = {} if by_loc is None else by_loc
    by_group: dict[str, int] = {}
    chosen: list[str] = []
    for row in order.itertuples():
        if len(chosen) >= n:
            break
        if by_loc.get(row.location, 0) >= per_location:
            continue
        if per_group is not None and by_group.get(row.group, 0) >= per_group:
            continue
        chosen.append(row.seq_key)
        by_loc[row.location] = by_loc.get(row.location, 0) + 1
        if per_group is not None:
            by_group[row.group] = by_group.get(row.group, 0) + 1
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--out", default="data/swg")
    ap.add_argument("--metadata", help="Use a local COCO-CT .json/.zip instead of downloading")
    ap.add_argument("--no-images", action="store_true", help="Write the manifest only")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dcfg, scfg = cfg["dataset"], cfg["sampling"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(scfg["seed"])

    meta_path = Path(args.metadata) if args.metadata else fetch_metadata(dcfg["metadata_url"], out)
    print("Parsing metadata...")
    meta = load_coco_ct(meta_path)
    images = build_image_table(
        meta,
        exclude_labels=dcfg["exclude_labels"],
        empty_label=dcfg.get("empty_label", "empty"),
        species_groups=dcfg.get("species_groups"),
    )
    del meta
    print(f"{len(images):,} images | gt counts: {images['gt'].value_counts().to_dict()}")

    images["seq_key"] = assign_events(images, cfg["events"]["max_gap_seconds"])
    seqs = sequence_table(images)
    animal_seqs = seqs[seqs["gt"] == GT_ANIMAL]
    empty_seqs = seqs[seqs["gt"] == GT_EMPTY]
    print(f"Usable sequences: {len(animal_seqs):,} animal, {len(empty_seqs):,} empty")
    print("Animal sequences by group:\n" + animal_seqs["group"].value_counts().to_string())

    per_loc = scfg["max_sequences_per_location"]
    # Each camera gets at most half its quota from animals, so empties from the
    # same cameras still fit; the shared counter enforces the total cap.
    loc_counts: dict[str, int] = {}
    animal_keys = capped_sample(
        animal_seqs,
        scfg["n_animal_sequences"],
        rng,
        max(1, per_loc // 2),
        scfg["max_sequences_per_group"],
        by_loc=loc_counts,
    )
    empty_keys = capped_sample(empty_seqs, scfg["n_empty_sequences"], rng, per_loc, by_loc=loc_counts)
    chosen = set(animal_keys) | set(empty_keys)
    if len(animal_keys) < scfg["n_animal_sequences"] or len(empty_keys) < scfg["n_empty_sequences"]:
        print(
            f"Note: caps limited the sample to {len(animal_keys)} animal / {len(empty_keys)} empty sequences. "
            "Raise max_sequences_per_location or max_sequences_per_group for more."
        )

    sample = images[images["seq_key"].isin(chosen)].copy()
    seq_gt = seqs.set_index("seq_key")
    sample["seq_gt"] = sample["seq_key"].map(seq_gt["gt"])
    sample["seq_group"] = sample["seq_key"].map(seq_gt["group"])
    base = dcfg["image_base_url"].rstrip("/") + "/"
    sample["url"] = base + sample["file_name"].str.lstrip("/")
    sample["local_path"] = [str(out / "images" / f) for f in sample["file_name"].str.lstrip("/")]
    sample["labels"] = sample["labels"].map(lambda xs: "|".join(xs))

    cols = [
        "image_id",
        "file_name",
        "url",
        "local_path",
        "location",
        "datetime",
        "seq_id",
        "frame_num",
        "seq_key",
        "labels",
        "gt",
        "group",
        "seq_gt",
        "seq_group",
    ]
    manifest = out / "manifest.csv"
    sample[cols].to_csv(manifest, index=False)
    n_anim = (sample["seq_gt"] == GT_ANIMAL).sum()
    print(
        f"Manifest: {manifest} | {len(sample):,} images, {len(chosen):,} sequences "
        f"({n_anim:,} images from animal sequences), {sample['location'].nunique()} locations"
    )

    if args.no_images:
        return 0

    failed = 0
    with requests.Session() as session, ThreadPoolExecutor(scfg["download_workers"]) as pool:
        jobs = [pool.submit(download, u, Path(p), session) for u, p in zip(sample["url"], sample["local_path"])]
        for job in tqdm(as_completed(jobs), total=len(jobs), desc="images"):
            failed += 0 if job.result() else 1
    print(f"Done. {len(jobs) - failed:,} images on disk, {failed:,} failed (re-run to retry).")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
