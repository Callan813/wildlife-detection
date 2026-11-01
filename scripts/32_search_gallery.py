"""Phase 4, step 4: queries the labels can't score, as contact sheets to judge by eye.

    python scripts/32_search_gallery.py
    python scripts/32_search_gallery.py --query "civet eating" --query "sambar in the rain"

For each query, the top 12 events from the val + test cameras (the same search the
dashboard runs), each shown as the frame that matched with its box, plus the dataset label
underneath. Writes reports/phase4/gallery/<query>.jpg and results.csv; the verdicts go in
reports/phase4/by_eye.md.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from ghats.config import load_config
from ghats.search_service import Search, text_encoder

QUERIES = [
    "animal drinking water",
    "animal close to the camera",
    "macaque climbing a tree",
    "wild pig digging in the ground",
    "animal running",
    "mother with young",
    "bird on the ground",
    "animal in the rain",
    "two wild pigs",
    "sambar at night",
    "a tiger",  # no tigers in this data: what does it return?
]
TILE = 360


def sheet(search: Search, res: dict, n: int) -> Image.Image:
    rows = (n + 3) // 4
    out = Image.new("RGB", (TILE * 4, rows * (TILE * 3 // 4 + 22)), "white")
    d = ImageDraw.Draw(out)
    truth = search.idx.events.set_index("seq_key")["truth"]
    for k, r in enumerate(res["results"][:n]):
        im = Image.open(search.frame_path[r["image_id"]])
        im.draft("RGB", (TILE * 2, TILE * 2))
        im = im.convert("RGB").resize((TILE, TILE * 3 // 4))
        if r["bbox"]:
            x1, y1, x2, y2 = r["bbox"]
            ImageDraw.Draw(im).rectangle(
                [x1 * im.width, y1 * im.height, x2 * im.width, y2 * im.height], outline="#ffd23f", width=3
            )
        x, y = (k % 4) * TILE, (k // 4) * (TILE * 3 // 4 + 22)
        out.paste(im, (x, y))
        night = " night" if r["night"] else ""
        d.text((x + 4, y + TILE * 3 // 4 + 4), f"{k + 1}. {truth[r['event']]}{night}  {r['location']}", fill="black")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase4.yaml")
    ap.add_argument("--query", action="append", help="default: the list in this script")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="reports/phase4/gallery")
    args = ap.parse_args()
    cfg = load_config(args.config)
    search = Search(cfg, text_encoder(Path(cfg["models"]["dir"]) / cfg["models"]["search"], 4))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    truth = search.idx.events.set_index("seq_key")["truth"]

    rows = []
    for q in args.query or QUERIES:
        res = search.search(q, args.n)
        slug = re.sub(r"[^a-z0-9]+", "_", q.lower()).strip("_")
        sheet(search, res, args.n).save(out / f"{slug}.jpg", quality=85)
        rq = res["query"]
        print(f"{q!r}: {res['matches']} match, ranked by {rq['ranked_by']}, rest={rq['rest']!r}")
        for k, r in enumerate(res["results"], 1):
            rows.append(
                {"query": q, "rank": k, "event": r["event"], "label": truth[r["event"]], "night": r["night"],
                 "location": r["location"], "score": r["score"], "image": f"{slug}.jpg"}
            )  # fmt: skip
    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
