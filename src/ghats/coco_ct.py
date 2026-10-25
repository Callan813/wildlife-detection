"""Read COCO Camera Traps metadata (the format every LILA camera-trap dataset uses)
and turn it into one row per image with clean ground truth."""

from __future__ import annotations

import io
import json
import zipfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

GT_ANIMAL = "animal"
GT_EMPTY = "empty"
GT_EXCLUDED = "excluded"
# LILA withholds images of people and sensitive sites under private/ (they 404).
PRIVATE_PREFIX = "private/"


def load_coco_ct(path: str | Path) -> dict[str, Any]:
    """Load COCO-CT metadata from a .json file or a .zip that contains one."""
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if n.endswith(".json")]
            if not names:
                raise ValueError(f"No .json file inside {path}")
            # The class-level file is the largest JSON in the archive.
            name = max(names, key=lambda n: zf.getinfo(n).file_size)
            with zf.open(name) as f:
                return json.load(io.TextIOWrapper(f, encoding="utf-8"))
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_image_table(
    meta: dict[str, Any],
    exclude_labels: Iterable[str],
    empty_label: str = "empty",
    species_groups: dict[str, str] | None = None,
) -> pd.DataFrame:
    """One row per image.

    Columns: image_id, file_name, location, datetime, seq_id, frame_num,
    labels (sorted list), gt (animal | empty | excluded), group.

    Rules:
      * An image with any excluded label (blurred, human, ...) is excluded.
      * An image whose only label is the empty label is empty.
      * An image with an animal label is "animal". If species_groups is given,
        animals outside it are excluded, so the sample stays focused on
        Ghats-relevant groups.
      * An image with no annotation at all is excluded (unknown).
      * A corrupt image, or one under private/ (not downloadable), is excluded.
    """
    exclude = {s.lower() for s in exclude_labels}
    empty_label = empty_label.lower()
    groups = {k.lower(): v for k, v in (species_groups or {}).items()}

    cat_name = {c["id"]: str(c["name"]).strip().lower() for c in meta.get("categories", [])}
    labels_by_image: dict[Any, set[str]] = defaultdict(set)
    for ann in meta.get("annotations", []):
        name = cat_name.get(ann.get("category_id"))
        if name is not None:
            labels_by_image[ann["image_id"]].add(name)

    rows = []
    for img in meta.get("images", []):
        labels = sorted(labels_by_image.get(img["id"], set()))
        gt, group = _classify(labels, exclude, empty_label, groups)
        if img.get("corrupt") or str(img["file_name"]).startswith(PRIVATE_PREFIX):
            gt, group = GT_EXCLUDED, None
        rows.append(
            {
                "image_id": img["id"],
                "file_name": img["file_name"],
                "location": str(img.get("location", "unknown")),
                "datetime": img.get("datetime"),
                "seq_id": img.get("seq_id"),
                "frame_num": img.get("frame_num"),
                "labels": labels,
                "gt": gt,
                "group": group,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # LILA timestamps vary in format across datasets (and sometimes within one).
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", format="mixed")
    return df


def _classify(labels: list[str], exclude: set[str], empty_label: str, groups: dict[str, str]) -> tuple[str, str | None]:
    if not labels:
        return GT_EXCLUDED, None
    if any(lbl in exclude for lbl in labels):
        return GT_EXCLUDED, None
    animal = [lbl for lbl in labels if lbl != empty_label]
    if not animal:
        return GT_EMPTY, None
    if groups:
        mapped = sorted({groups[lbl] for lbl in animal if lbl in groups})
        if len(mapped) != 1 or any(lbl not in groups for lbl in animal):
            # Unmapped species or a multi-group frame: keep it out of a clean sample.
            return GT_EXCLUDED, None
        return GT_ANIMAL, mapped[0]
    return GT_ANIMAL, animal[0]
