"""Map off-the-shelf classifier labels onto our species groups.

SpeciesNet names ~2,500 taxa and the Central India model 40 classes; we have 16
groups. Each model's class probabilities are summed into our groups (plus
"other" for everything we do not use), so every model ends up with the same
columns as our own classifier's predictions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

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
