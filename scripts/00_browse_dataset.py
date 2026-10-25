"""Phase 1, step 0: look at the dataset before modelling it.

    python scripts/00_browse_dataset.py

Reads the COCO Camera Traps JSON and writes outputs/dataset_browser.html: pick a
label on the left and see random trigger events (bursts) for it. Photos load
straight from LILA storage, so nothing is downloaded. Only public/ images are
shown: the private/ ones (people, sensitive sites) return 404.

Standard library only, so it runs before the project dependencies are installed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_BASE = "https://storage.googleapis.com/public-datasets-lila/swg-camera-traps/"
MAX_FRAMES = 6  # frames kept per burst; long bursts are trimmed


def load_groups(config: Path) -> dict[str, str]:
    """LILA label -> Ghats group from the config, if PyYAML is available."""
    try:
        import yaml
    except ImportError:
        print("PyYAML not installed: Ghats groups will not be shown.")
        return {}
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {k.lower(): v for k, v in cfg["dataset"].get("species_groups", {}).items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", default="data/swg/swg_camera_traps.1.1.json")
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--out", default="outputs/dataset_browser.html")
    ap.add_argument("--per-label", type=int, default=30, help="bursts sampled per label")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Reading {args.metadata} (about a minute for SWG)...")
    with open(args.metadata, "r", encoding="utf-8") as f:
        meta = json.load(f)

    cat_name = {c["id"]: c["name"] for c in meta["categories"]}
    labels_by_image: dict[str, set[str]] = defaultdict(set)
    for ann in meta["annotations"]:
        labels_by_image[ann["image_id"]].add(cat_name[ann["category_id"]])

    n_all: Counter[str] = Counter()
    n_public: Counter[str] = Counter()
    seqs: dict[str, dict] = {}
    for img in meta["images"]:
        labels = labels_by_image.get(img["id"], set())
        n_all.update(labels)
        if not img["file_name"].startswith("public/") or img.get("corrupt"):
            continue
        n_public.update(labels)
        seq = seqs.setdefault(
            img["seq_id"],
            {"loc": img.get("location"), "country": img["file_name"].split("/")[1], "labels": set(), "frames": []},
        )
        seq["labels"] |= labels
        seq["frames"].append((img.get("frame_num") or 0, img["file_name"], img.get("datetime") or ""))
    del meta, labels_by_image

    seqs_by_label: dict[str, list[str]] = defaultdict(list)
    for sid, seq in seqs.items():
        for lbl in seq["labels"]:
            seqs_by_label[lbl].append(sid)

    rng = random.Random(args.seed)
    groups = load_groups(Path(args.config))
    out_labels = []
    for lbl, total in n_all.most_common():
        chosen = seqs_by_label.get(lbl, [])
        chosen = rng.sample(chosen, min(args.per_label, len(chosen)))
        bursts = []
        for sid in chosen:
            seq = seqs[sid]
            frames = sorted(seq["frames"])
            bursts.append(
                {
                    "loc": seq["loc"],
                    "country": seq["country"],
                    "time": frames[0][2][:16],
                    "labels": sorted(seq["labels"]),
                    "n": len(frames),
                    "frames": [f for _, f, _ in frames[:MAX_FRAMES]],
                }
            )
        out_labels.append(
            {
                "name": lbl,
                "group": groups.get(lbl.lower()),
                "all": total,
                "public": n_public[lbl],
                "seqs": len(seqs_by_label.get(lbl, [])),
                "bursts": bursts,
            }
        )

    data = json.dumps({"base": IMAGE_BASE, "labels": out_labels}).replace("</", "<\\/")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(PAGE.replace("__DATA__", data), encoding="utf-8")
    print(f"Wrote {out} ({len(out_labels)} labels). Open it in your browser.")
    return 0


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SWG Dataset Browser</title>
<style>
:root { --bg:#f6f5f1; --panel:#fff; --ink:#1d1d1b; --muted:#6b6b66; --line:#dddcd5; --accent:#2f6b3a; --sel:#e3eee4; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#151614; --panel:#1f201d; --ink:#e9e8e3; --muted:#9a9a93; --line:#33342f; --accent:#7fc28a; --sel:#26352a; }
}
* { box-sizing: border-box; }
body { margin:0; font:14px/1.4 system-ui, sans-serif; background:var(--bg); color:var(--ink); display:flex; height:100vh; }
aside { width:300px; flex:none; border-right:1px solid var(--line); display:flex; flex-direction:column; background:var(--panel); }
aside header { padding:12px; border-bottom:1px solid var(--line); }
aside h1 { font-size:15px; margin:0 0 8px; }
aside input { width:100%; padding:6px 8px; border:1px solid var(--line); border-radius:6px; background:var(--bg); color:var(--ink); }
aside label { display:block; margin-top:8px; color:var(--muted); font-size:12px; }
#list { overflow:auto; flex:1; }
.item { padding:6px 12px; cursor:pointer; display:flex; justify-content:space-between; gap:8px; border-bottom:1px solid var(--line); }
.item:hover { background:var(--bg); } .item.on { background:var(--sel); }
.item small { color:var(--muted); white-space:nowrap; }
.tag { font-size:11px; color:var(--accent); }
main { flex:1; overflow:auto; padding:16px; }
main h2 { margin:0 0 4px; } .sub { color:var(--muted); margin-bottom:14px; }
.grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(320px, 1fr)); gap:14px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; }
.card img { width:100%; aspect-ratio:4/3; object-fit:cover; display:block; background:#000; cursor:zoom-in; }
.strip { display:flex; gap:4px; padding:6px; }
.strip button { flex:1; padding:3px 0; border:1px solid var(--line); border-radius:4px; background:var(--bg); color:var(--ink); cursor:pointer; font-size:12px; }
.strip button.on { background:var(--accent); color:var(--panel); border-color:var(--accent); }
.meta { padding:0 8px 8px; font-size:12px; color:var(--muted); }
.meta b { color:var(--ink); }
@media (max-width: 700px) { body { flex-direction:column; height:auto; } aside { width:100%; max-height:45vh; } }
</style></head><body>
<aside><header>
  <h1>SWG Camera Traps</h1>
  <input id="q" placeholder="Filter labels... (e.g. civet)">
  <label><input type="checkbox" id="ghats" style="width:auto"> Only labels mapped to a Ghats group</label>
</header><div id="list"></div></aside>
<main id="main"></main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const list = document.getElementById('list'), main = document.getElementById('main');
const q = document.getElementById('q'), ghats = document.getElementById('ghats');
let current = D.labels[0].name;
const fmt = n => n.toLocaleString();

function renderList() {
  const s = q.value.toLowerCase();
  list.innerHTML = '';
  for (const L of D.labels) {
    if (s && !L.name.toLowerCase().includes(s)) continue;
    if (ghats.checked && !L.group) continue;
    const el = document.createElement('div');
    el.className = 'item' + (L.name === current ? ' on' : '');
    el.innerHTML = `<span>${L.name}${L.group ? `<br><span class="tag">→ ${L.group}</span>` : ''}</span><small>${fmt(L.all)}</small>`;
    el.onclick = () => { current = L.name; renderList(); renderMain(); };
    list.appendChild(el);
  }
}

function renderMain() {
  const L = D.labels.find(l => l.name === current);
  main.scrollTop = 0;
  main.innerHTML = `<h2>${L.name}</h2>
    <div class="sub">${fmt(L.all)} images in total · ${fmt(L.public)} public · ${fmt(L.seqs)} public bursts
    ${L.group ? ' · Ghats group: <b>' + L.group + '</b>' : ''} · showing ${L.bursts.length} random bursts</div>
    ${L.bursts.length ? '' : '<p>No public images for this label (all private).</p>'}<div class="grid"></div>`;
  const grid = main.querySelector('.grid');
  for (const b of L.bursts) {
    const card = document.createElement('div');
    card.className = 'card';
    const img = document.createElement('img');
    img.loading = 'lazy';
    const show = i => {
      img.src = D.base + b.frames[i];
      card.querySelectorAll('.strip button').forEach((x, j) => x.classList.toggle('on', j === i));
    };
    img.onclick = () => window.open(img.src, '_blank');
    card.appendChild(img);
    const strip = document.createElement('div');
    strip.className = 'strip';
    b.frames.forEach((_, i) => {
      const btn = document.createElement('button');
      btn.textContent = 'frame ' + (i + 1);
      btn.onclick = () => show(i);
      strip.appendChild(btn);
    });
    card.appendChild(strip);
    const other = b.labels.filter(x => x !== L.name);
    card.insertAdjacentHTML('beforeend', `<div class="meta"><b>${b.loc}</b> (${b.country}) · ${b.time} UTC ·
      ${b.n} frame${b.n > 1 ? 's' : ''}${other.length ? '<br>also labelled: <b>' + other.join(', ') + '</b>' : ''}</div>`);
    grid.appendChild(card);
    show(0);
  }
}

q.oninput = renderList;
ghats.onchange = renderList;
renderList(); renderMain();
</script></body></html>
"""

if __name__ == "__main__":
    sys.exit(main())
