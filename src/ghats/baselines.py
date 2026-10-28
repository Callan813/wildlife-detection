"""Map off-the-shelf classifier labels onto our species groups, and cut crops the way they expect.

SpeciesNet names ~2,500 taxa and the Central India model 40 classes; we have 16
groups. Each model's class probabilities are summed into our groups (plus
"other" for everything we do not use), so every model ends up with the same
columns as our own classifier's predictions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from PIL import Image

OTHER = "other"
_LEVELS = ("class", "order", "family", "genus", "species")


def speciesnet_group(label: str, rules: Sequence[Mapping[str, str]]) -> str:
    """Group for one SpeciesNet label "uuid;class;order;family;genus;species;common name".

    Each rule names one level (class/order/family/genus/species/common) and a value;
    the first rule that matches decides the group.
    """
    parts = label.strip().split(";")
    if len(parts) != 7:
        raise ValueError(f"not a SpeciesNet label: {label!r}")
    fields = dict(zip(_LEVELS, (p.lower() for p in parts[1:6])))
    fields["common"] = parts[6].lower()
    for rule in rules:
        level = next(k for k in rule if k != "group")
        if fields.get(level) == str(rule[level]).lower():
            return rule["group"]
    return OTHER


def group_probs(probs: np.ndarray, class_groups: Sequence[str], groups: Sequence[str]) -> np.ndarray:
    """Sum per-class probabilities (n, n_classes) into ``groups`` plus a last "other" column."""
    cols = {g: k for k, g in enumerate(groups)}
    index = np.array([cols.get(g, len(groups)) for g in class_groups])
    out = np.zeros((probs.shape[0], len(groups) + 1), dtype=probs.dtype)
    np.add.at(out.T, index, probs.T)
    return out


SPECIESNET_SIZE = 480


class TightCrops:
    """Map-style dataset: the detector box cut from the original frame, preprocessed for one model."""

    def __init__(self, paths: list[str], bboxes: list[list[float]], preprocess):
        self.paths = paths
        self.bboxes = bboxes
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB")
            x1, y1, x2, y2 = self.bboxes[i]
            box = (int(x1 * im.width), int(y1 * im.height), int(x2 * im.width), int(y2 * im.height))
            if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                box = (0, 0, im.width, im.height)  # degenerate box: fall back to the frame
            return self.preprocess(im.crop(box))


def speciesnet_preprocess(crop: Image.Image):
    """SpeciesNet's own pipeline: resize to 480 without antialiasing, uint8 round trip, NHWC in [0, 1]."""
    import torch
    import torchvision.transforms.functional as TF

    x = TF.convert_image_dtype(TF.pil_to_tensor(crop), torch.float32)
    x = TF.resize(x, [SPECIESNET_SIZE, SPECIESNET_SIZE], antialias=False)
    x = TF.convert_image_dtype(x, torch.uint8)
    return x.permute(1, 2, 0).float() / 255.0
