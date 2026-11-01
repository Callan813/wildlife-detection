"""Phase 4: the search behind scripts/31_search_server.py (and 32_search_gallery.py).

Wraps ghats.search.Index with what a client needs per result: the frame and box that
matched, the likely species, time and camera, and image bytes for the indexed photos only.
"""

from __future__ import annotations

import io
import json
import threading
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ghats.search import Index, Query, load_index, parse_query, query_prompt

FRAME_WIDTH = 1280


class Search:
    def __init__(self, cfg: dict, encode):
        self.cfg, self.encode = cfg, encode
        name = cfg["models"]["search"]
        self.idx: Index = load_index(cfg, name)
        self.vocab = {g: v["words"] for g, v in cfg["species"].items()}
        d = cfg["data"]
        crops = pd.read_csv(d["crops"], dtype={"image_id": str}).set_index("crop_id")
        self.crops = crops.loc[self.idx.crop_ids, ["image_id", "bbox", "conf"]]
        self.crops["bbox"] = self.crops["bbox"].map(json.loads)
        fr = pd.read_csv(d["manifest"], dtype={"image_id": str}).set_index("image_id")
        self.frame_path = fr.loc[self.idx.frame_ids, "local_path"].to_dict()
        self.crop_dir = Path(d["crop_dir"])
        probs = pd.read_csv(d["predictions"]).set_index("crop_id")
        self.crop_probs = probs.loc[self.idx.crop_ids, [f"p_{g}" for g in self.idx.groups] + ["p_background"]]
        # Boxes the detector is fairly sure of, per frame: for "two wild pigs", show the frame with most.
        sure = self.crops["conf"].to_numpy() >= cfg["search"]["min_box_conf"]
        per_image = pd.Series(sure).groupby(self.crops["image_id"].to_numpy()).sum()
        self.image_boxes = per_image.loc[self.crops["image_id"]].to_numpy()
        ev = self.idx.events
        self.first_frame = {}
        for i, e in zip(self.idx.frame_event, self.idx.frame_ids):
            self.first_frame.setdefault(int(i), str(e))
        self.event_row = pd.Series(np.arange(len(ev)), index=ev["seq_key"])

    def best_crop(self, i: int, vec: np.ndarray | None, groups: list[str], min_count: int = 0) -> int | None:
        """Position in crop_ids of the crop that best explains event i's score."""
        rows = np.flatnonzero(self.idx.crop_event == i)
        if vec is not None:
            rows = rows[self.idx.crop_ok[rows]]  # the crops the score could have come from
        if min_count and len(rows):
            rows = rows[self.image_boxes[rows] == self.image_boxes[rows].max()]
        if not len(rows):
            return None
        if vec is not None:
            s = self.idx.crop_vecs[rows].astype(np.float32) @ vec
        elif groups:
            s = self.crop_probs.iloc[rows][[f"p_{g}" for g in groups]].max(axis=1).to_numpy()
        else:
            s = 1.0 - self.crop_probs.iloc[rows]["p_background"].to_numpy()
        return int(rows[np.argmax(s)])

    def best_frame(self, i: int, vec: np.ndarray | None) -> str:
        rows = np.flatnonzero(self.idx.frame_event == i)
        if vec is None:
            return self.first_frame[i]
        return str(self.idx.frame_ids[rows[np.argmax(self.idx.frame_vecs[rows].astype(np.float32) @ vec)]])

    def result(self, i: int, score: float, vec, groups, min_count: int = 0) -> dict:
        e = self.idx.events.iloc[i]
        species = max(self.idx.groups, key=lambda g: e[f"p_{g}"])
        # Show the best box when there is one (it points the eye at the animal), else the best frame.
        c = self.best_crop(i, vec, groups, min_count)
        crop = None if c is None else self.crops.iloc[c]
        image_id = str(crop["image_id"]) if crop is not None else self.best_frame(i, vec)
        return {
            "event": e["seq_key"],
            "location": e["location"],
            "start": e["start"].isoformat(),
            "night": bool(self.idx.night()[i]),
            "frames": int(e["frames"]),
            "score": round(float(score), 4),
            "p_animal": round(float(e["p_animal"]), 3),
            "species": species if e["p_animal"] >= self.idx.animal_min_prob else None,
            "species_prob": round(float(e[f"p_{species}"]), 3),
            "n_boxes": int(e["n_boxes"]),
            "image_id": image_id,
            "frame_url": f"/images/search/frame/{image_id}",
            "crop_url": None if crop is None else f"/images/search/crop/{self.idx.crop_ids[c]}",
            "bbox": None if crop is None else crop["bbox"],
        }

    def search(self, text: str, k: int) -> dict:
        q = parse_query(text, self.vocab)
        semantic = bool(q.rest) or not q.has_filters
        vec = None
        if semantic:
            vec = self.encode([query_prompt(q)])[0]
            scores = np.where(self.idx.mask(q), self.idx.similarity(vec), -np.inf)
        else:
            scores = self.idx.score(q, self.encode)
        order = [i for i in np.argsort(-scores, kind="stable")[:k] if np.isfinite(scores[i])]
        return {
            "query": {**asdict(q), "ranked_by": "meaning" if semantic else "species model"},
            "matches": int(np.isfinite(scores).sum()),
            "results": [self.result(int(i), scores[i], vec, q.species, q.min_count) for i in order],
        }

    def similar(self, event: str, k: int) -> dict:
        if event not in self.event_row:
            raise KeyError(event)
        i = int(self.event_row[event])
        c = self.best_crop(i, None, [])
        rows = np.flatnonzero(self.idx.frame_event == i)
        vec = (self.idx.crop_vecs[c] if c is not None else self.idx.frame_vecs[rows[0]]).astype(np.float32)
        scores = self.idx.similarity(vec)
        scores[i] = -np.inf
        order = np.argsort(-scores, kind="stable")[:k]
        return {
            "query": asdict(Query(text=f"like {event}")) | {"ranked_by": "looks like"},
            "matches": len(scores) - 1,
            "results": [self.result(int(j), scores[j], vec, []) for j in order],
        }

    def image(self, kind: str, key: str) -> bytes | None:
        if kind == "crop":
            if key not in self.crops.index:
                return None
            return (self.crop_dir / f"{key}.jpg").read_bytes()
        path = self.frame_path.get(key)
        if path is None:
            return None
        im = Image.open(path)
        im.draft("RGB", (FRAME_WIDTH, FRAME_WIDTH))
        im = im.convert("RGB")
        if im.width > FRAME_WIDTH:
            im = im.resize((FRAME_WIDTH, round(im.height * FRAME_WIDTH / im.width)), Image.BILINEAR)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
        return buf.getvalue()


def text_encoder(mdir: Path, threads: int):
    import open_clip
    import torch

    torch.set_num_threads(threads)
    model, _, _ = open_clip.create_model_and_transforms(f"local-dir:{mdir}")
    model.visual = None  # only the text side is needed
    tok = open_clip.get_tokenizer(f"local-dir:{mdir}")
    model.eval()

    lock = threading.Lock()  # the server is threaded; one forward pass at a time

    @torch.no_grad()
    def encode(texts: list[str]) -> np.ndarray:
        with lock:
            f = model.encode_text(tok(texts))
        return (f / f.norm(dim=-1, keepdim=True)).numpy()

    return encode
