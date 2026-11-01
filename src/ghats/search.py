"""Phase 4: search the camera-trap photos in plain English.

A query like "two wild pigs at night on camera 12" is split in two:

  filters   the parts the metadata already answers exactly: species (from the Phase 2
            probe), night or day (capture hour), empty or not, how many boxes, camera,
            year, month
  meaning   the rest ("drinking", "close to the camera", "in the rain"), matched by
            comparing a text embedding with the image embeddings of every frame and crop

Results are trigger events, not single frames: an event scores as well as its best frame
or crop, the same unit as the Phase 1 and 2 evaluations. Everything here is numpy and
pandas; the text encoder is passed in, so tests don't need a model.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

NIGHT_WORDS = ["night time", "night-time", "nighttime", "night", "nocturnal", "dark", "infrared", "ir"]
DAY_WORDS = ["day time", "daytime", "daylight", "day", "sunlight", "sunny"]
EMPTY_WORDS = ["no animals", "no animal", "nothing", "empty", "blank"]
ANIMAL_WORDS = ["animals", "animal", "anything", "wildlife"]
# Words that only make sense with an animal in the photo; they stay in the text for the embeddings.
ANIMAL_HINTS = {"mother", "baby", "babies", "young", "cub", "cubs", "calf", "juvenile", "eating", "drinking",
                "running", "sleeping", "fighting", "climbing", "digging", "walking"}  # fmt: skip
COUNT_WORDS = {
    "two": 2, "2": 2, "pair": 2, "pair of": 2, "couple": 2, "couple of": 2,
    "three": 3, "3": 3, "four": 4, "4": 4,
    "group": 2, "group of": 2, "herd": 2, "herd of": 2, "troop": 2, "troop of": 2,
    "several": 2, "multiple": 2, "many": 2,
}  # fmt: skip
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]  # fmt: skip
FILLER = {"a", "an", "the", "of", "at", "in", "on", "with", "and", "or", "more", "photo", "photos", "picture",
          "pictures", "image", "images", "show", "me", "find", "all", "any", "some", "from", "camera", "trap",
          "by", "taken", "during", "shot", "shots", "seen", "only", "please", "there", "is", "are", "where"}  # fmt: skip


@dataclass
class Query:
    text: str
    species: list[str] = field(default_factory=list)
    time: str | None = None  # "night" or "day"
    empty: bool = False
    animal: bool = False
    min_count: int = 0
    cameras: list[str] = field(default_factory=list)
    years: list[int] = field(default_factory=list)
    months: list[int] = field(default_factory=list)
    rest: str = ""  # what the filters didn't use up: left to the embeddings

    @property
    def has_filters(self) -> bool:
        return bool(
            self.species or self.time or self.empty or self.animal or self.min_count
            or self.cameras or self.years or self.months
        )  # fmt: skip


def _take(text: str, phrases: list[str]) -> tuple[str, list[str]]:
    """Remove whole-word matches of phrases (longest first) from text; return the rest and the hits."""
    hits = []
    for p in sorted(phrases, key=len, reverse=True):
        pat = re.compile(rf"(?<![\w-]){re.escape(p)}(?![\w-])")
        if pat.search(text):
            hits.append(p)
            text = pat.sub(" ", text)
    return text, hits


def parse_query(text: str, vocab: dict[str, list[str]]) -> Query:
    """Pull the structured parts out of a query. ``vocab`` maps species group -> words."""
    q = Query(text=text.strip())
    t = " " + re.sub(r"[^\w\s-]", " ", text.lower()) + " "

    for m in re.finditer(r"\b(?:loc_|camera\s*|cam\s*)(\d{1,4})\b", t):
        q.cameras.append(f"loc_{int(m.group(1)):04d}")
    t = re.sub(r"\b(?:loc_|camera\s*|cam\s*)\d{1,4}\b", " ", t)
    for m in re.finditer(r"\b(20\d\d)\b", t):
        q.years.append(int(m.group(1)))
    t = re.sub(r"\b20\d\d\b", " ", t)
    t, hits = _take(t, MONTHS)
    q.months = sorted(MONTHS.index(h) + 1 for h in hits)

    word_to_group = {}
    for g, ws in vocab.items():
        for w in ws:
            w = w.lower()
            word_to_group.setdefault(w, g)
            if not w.endswith("s"):
                word_to_group.setdefault(w + "s", g)  # plain plurals: "wild pigs", "civets"
    t, hits = _take(t, list(word_to_group))
    q.species = sorted({word_to_group[h] for h in hits})

    t, hits = _take(t, EMPTY_WORDS)
    q.empty = bool(hits)
    t, hits = _take(t, NIGHT_WORDS)
    night = bool(hits)
    t, hits = _take(t, DAY_WORDS)
    if night != bool(hits):  # "day and night" means either
        q.time = "night" if night else "day"
    t, hits = _take(t, list(COUNT_WORDS))
    q.min_count = max((COUNT_WORDS[h] for h in hits), default=0)
    t, hits = _take(t, ANIMAL_WORDS)
    q.animal = bool(hits) and not q.empty

    q.rest = " ".join(w for w in t.split() if w not in FILLER)
    if not q.empty and ANIMAL_HINTS & set(q.rest.split()):
        q.animal = True
    return q


def species_prompts(group: str, words: list[str], taxon: str | None, style: str = "both") -> list[str]:
    """Text prompts for one species group. ``style``: "common", "taxon" or "both"."""
    out = []
    if style in ("common", "both"):
        out += [f"a camera trap photo of a {words[0]}.", f"a photo of a {words[0]}."]
    if style in ("taxon", "both") and taxon:
        out += [f"a photo of {taxon}."]
    return out or [f"a photo of a {group.replace('_', ' ')}."]


def query_prompt(q: Query) -> str:
    return f"a camera trap photo of {q.text.strip().rstrip('.')}."


def event_max(values: np.ndarray, event_idx: np.ndarray, n_events: int) -> np.ndarray:
    """Max of ``values`` per event (-inf for events with none)."""
    out = np.full(n_events, -np.inf, dtype=np.float32)
    np.maximum.at(out, event_idx, values.astype(np.float32))
    return out


def average_precision(scores: np.ndarray, relevant: np.ndarray) -> float:
    """AP of a ranking by ``scores`` (higher first); NaN if nothing is relevant."""
    relevant = np.asarray(relevant, dtype=bool)
    if not relevant.any():
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    rel = relevant[order]
    hits = np.cumsum(rel)
    return float((hits[rel] / (np.flatnonzero(rel) + 1)).mean())


def precision_at(scores: np.ndarray, relevant: np.ndarray, k: int) -> float:
    order = np.argsort(-scores, kind="stable")[:k]
    return float(np.asarray(relevant, dtype=bool)[order].mean()) if len(order) else float("nan")


@dataclass
class Index:
    """Events plus the embeddings of their frames and crops.

    ``events`` is a pandas DataFrame, one row per trigger event, with columns: seq_key,
    location, start (Timestamp), hour, colour, n_boxes (most boxes >= min conf in one
    frame), p_animal (best box), p_<group> per species group (best box).
    ``frame_vecs`` / ``crop_vecs`` are L2-normalised; ``frame_event`` / ``crop_event`` give the
    row in ``events`` for each. Only crops with detector confidence >= ``crop_min_conf`` are
    matched against text: the Phase 2 crops go down to 0.02, and a patch of leaves can
    resemble "a bird" better than the real bird in the next box.
    """

    events: object
    frame_vecs: np.ndarray
    frame_event: np.ndarray
    crop_vecs: np.ndarray
    crop_event: np.ndarray
    groups: list[str]
    night_hours: tuple[int, int] = (18, 6)
    species_min_prob: float = 0.5
    animal_min_prob: float = 0.273
    frame_ids: np.ndarray | None = None
    crop_ids: np.ndarray | None = None
    crop_conf: np.ndarray | None = None
    crop_min_conf: float = 0.0

    @property
    def crop_ok(self) -> np.ndarray:
        if self.crop_conf is None:
            return np.ones(len(self.crop_event), dtype=bool)
        return self.crop_conf >= self.crop_min_conf

    def similarity(self, vec: np.ndarray, use: str = "both") -> np.ndarray:
        """Per-event cosine similarity: best frame, best crop, or the best of either."""
        n = len(self.events)
        v = vec.astype(np.float32)
        out = np.full(n, -np.inf, dtype=np.float32)
        if use in ("frames", "both"):
            out = np.maximum(out, event_max(self.frame_vecs.astype(np.float32) @ v, self.frame_event, n))
        if use in ("crops", "both"):
            ok = self.crop_ok
            sims = self.crop_vecs[ok].astype(np.float32) @ v
            out = np.maximum(out, event_max(sims, self.crop_event[ok], n))
        return out

    def night(self) -> np.ndarray:
        h = self.events["hour"].to_numpy()
        a, b = self.night_hours
        return (h >= a) | (h < b)

    def mask(self, q: Query) -> np.ndarray:
        ev = self.events
        m = np.ones(len(ev), dtype=bool)
        if q.species:
            p = np.max([ev[f"p_{g}"].to_numpy() for g in q.species], axis=0)
            m &= p >= self.species_min_prob
        if q.time == "night":
            m &= self.night()
        elif q.time == "day":
            m &= ~self.night()
        if q.empty:
            m &= ev["p_animal"].to_numpy() < self.animal_min_prob
        if q.animal or q.min_count:
            m &= ev["p_animal"].to_numpy() >= self.animal_min_prob
        if q.min_count:
            m &= ev["n_boxes"].to_numpy() >= q.min_count
        if q.cameras:
            m &= ev["location"].isin(q.cameras).to_numpy()
        if q.years:
            m &= ev["start"].dt.year.isin(q.years).to_numpy()
        if q.months:
            m &= ev["start"].dt.month.isin(q.months).to_numpy()
        return m

    def score(self, q: Query, encode_text: Callable[[list[str]], np.ndarray]) -> np.ndarray:
        """Score every event for ``q``; events the filters rule out get -inf.

        With words left over after the filters ("drinking", "close up"), events are ranked by
        how well the whole query text matches their best frame or crop. Without any
        ("civets at night"), by the probe's probability for the species asked for, or for
        any animal (most likely first; for "empty", least likely first).
        """
        m = self.mask(q)
        ev = self.events
        if q.rest or not q.has_filters:
            s = self.similarity(encode_text([query_prompt(q)])[0])
        elif q.species:
            s = np.max([ev[f"p_{g}"].to_numpy() for g in q.species], axis=0).astype(np.float32)
        elif q.empty:
            s = -ev["p_animal"].to_numpy(dtype=np.float32)
        else:
            s = ev["p_animal"].to_numpy(dtype=np.float32)
        return np.where(m, s, -np.inf)

    def search(self, q: Query, encode_text, k: int = 24) -> list[int]:
        s = self.score(q, encode_text)
        order = np.argsort(-s, kind="stable")[:k]
        return [int(i) for i in order if np.isfinite(s[i])]


def load_index(cfg: dict, model: str) -> Index:
    """Build the Index for the val + test cameras from the Phase 2 files and 29_embed.py's output."""
    from pathlib import Path

    import pandas as pd

    d, s = cfg["data"], cfg["search"]
    groups = list(cfg["species"])
    split = pd.read_csv(d["splits"]).set_index("location")["split"]
    fr = pd.read_csv(d["manifest"], dtype={"image_id": str})
    fr["split"] = fr["location"].map(split)
    fr = fr[fr["split"].isin(d["index_splits"])]
    fr["datetime"] = pd.to_datetime(fr["datetime"], utc=True).dt.tz_localize(None)

    emb = Path(cfg["embeddings"])
    fz = np.load(emb / f"{model}_frames.npz")
    cz = np.load(emb / f"{model}_crops.npz")
    colour = pd.Series(fz["colour"], index=fz["ids"])
    fr["colour"] = fr["image_id"].map(colour)

    first = fr.sort_values(["seq_key", "datetime"]).groupby("seq_key")
    events = first.agg(
        location=("location", "first"),
        split=("split", "first"),
        start=("datetime", "min"),
        frames=("image_id", "size"),
        colour=("colour", "median"),
        truth=("seq_group", "first"),
        gt=("seq_gt", "first"),
    ).reset_index()
    events["truth"] = events["truth"].where(events["gt"] == "animal", "empty")
    events["hour"] = events["start"].dt.hour

    p = pd.read_csv(d["predictions"], dtype={"image_id": str})
    p = p[p["seq_key"].isin(events["seq_key"])]
    p["p_animal"] = 1.0 - p["p_background"]
    best = p.groupby("seq_key")[["p_animal"] + [f"p_{g}" for g in groups]].max()
    boxes = p[p["conf"] >= s["min_box_conf"]].groupby(["seq_key", "image_id"]).size().groupby("seq_key").max()
    events = events.merge(best, left_on="seq_key", right_index=True, how="left")
    events["n_boxes"] = events["seq_key"].map(boxes)
    events = events.fillna({c: 0.0 for c in ["p_animal", "n_boxes"] + [f"p_{g}" for g in groups]})
    events["n_boxes"] = events["n_boxes"].astype(int)

    row = pd.Series(np.arange(len(events)), index=events["seq_key"])
    frame_seq = fr.set_index("image_id")["seq_key"]
    crop_seq = p.set_index("crop_id")["seq_key"]
    crop_conf = p.set_index("crop_id")["conf"]
    fids, cids = fz["ids"], cz["ids"]
    return Index(
        events=events,
        frame_vecs=fz["vecs"],
        frame_event=row[frame_seq.loc[fids].to_numpy()].to_numpy(),
        crop_vecs=cz["vecs"],
        crop_event=row[crop_seq.loc[cids].to_numpy()].to_numpy(),
        groups=groups,
        night_hours=tuple(s["night_hours"]),
        species_min_prob=s["species_min_prob"],
        frame_ids=fids,
        crop_ids=cids,
        crop_conf=crop_conf.loc[cids].to_numpy(dtype=np.float32),
        crop_min_conf=s.get("crop_min_conf", 0.0),
    )
