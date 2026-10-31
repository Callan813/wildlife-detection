"""Phase 6: run the camera-side pipeline on photos, a video or a webcam.

    python scripts/26_live.py --source path/to/frames/        # camera-trap photos
    python scripts/26_live.py --source clip.mp4               # a video, sampled at 2 fps
    python scripts/26_live.py --source 0 --show               # webcam 0, with a live window
    python scripts/26_live.py --source frames/ --variant fp32 --threads 2

For every frame: detector -> animal boxes -> species model + probe head. Frames are grouped
into events (camera-trap bursts by time gap; video by runs of frames with an animal), and
each event is kept or dropped with the Phase 2 operating point (configs/phase6.yaml). For a
kept event, the camera would send one padded crop of its best box plus a line of metadata,
instead of every frame.

Writes to --out (default outputs/phase6/live/<source name>):
  events.jsonl   one line per event: kept, species, P(animal), best frame and box
  frames.jsonl   one line per frame: boxes with detector conf, species, P(animal), timings
  crops/         the crop a camera would send for each kept event
  summary.json   counts, timings and bytes sent vs sending every frame
  sheet.jpg      with --sheet: each event's deciding frame, box and verdict, in a grid

Only numpy, Pillow and onnxruntime are needed for photos; video and webcam add OpenCV.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from ghats.config import load_config
from ghats.live import AnimalSpans, Event, EventGrouper, FrameResult, Pipeline

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def taken_at(path: Path, im: Image.Image) -> float:
    """Capture time in seconds: EXIF DateTimeOriginal / DateTime if present, else file time."""
    exif = im.getexif()
    raw = exif.get_ifd(0x8769).get(36867) or exif.get(306)
    if raw:
        try:
            return datetime.strptime(str(raw).strip(), "%Y:%m:%d %H:%M:%S").timestamp()
        except ValueError:
            pass
    return path.stat().st_mtime


def photos(src: Path) -> Iterator[tuple[str, float, Image.Image, int]]:
    files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXT)
    timed = []
    for p in files:
        with Image.open(p) as im:
            timed.append((taken_at(p, im), p))
    for t, p in sorted(timed):
        with Image.open(p) as im:
            yield str(p.relative_to(src) if src.is_dir() else p.name), t, im.convert("RGB"), p.stat().st_size


def video(src: str | int, fps: float, max_seconds: float | None) -> Iterator[tuple[str, float, Image.Image, int]]:
    import cv2

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"can't open {src!r}")
    live = isinstance(src, int)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / fps)) if not live else 1
    start, i, last = time.monotonic(), 0, -1e9
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            t = time.monotonic() - start if live else i / src_fps
            i += 1
            if (live and t - last < 1 / fps) or (not live and (i - 1) % step):
                continue
            last = t
            if max_seconds is not None and t > max_seconds:
                break
            ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])  # what sending the frame costs
            im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            yield f"{int(t // 60)}:{t % 60:04.1f}", t, im, len(jpg) if ok else 0
    finally:
        cap.release()


def crop_bytes(im: Image.Image, bbox: list[float], c: dict) -> tuple[bytes, Image.Image]:
    from ghats.crops import crop_box

    out = im.crop(crop_box(bbox, im.width, im.height, c["pad"], 32))
    out.thumbnail((c["max_side"], c["max_side"]), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=c["jpeg_quality"])
    return buf.getvalue(), out


TILE = (400, 300)


def tile(im: Image.Image | None, s: dict, n: int) -> Image.Image:
    """One event for --sheet: its deciding frame, the box (yellow if kept), and the verdict."""
    from PIL import ImageDraw

    out = Image.new("RGB", TILE, "black")
    if im is not None:
        th = im.copy()
        th.thumbnail((TILE[0], TILE[1] - 26))
        ox, oy = (TILE[0] - th.width) // 2, 0
        out.paste(th, (ox, oy))
        d = ImageDraw.Draw(out)
        if s["best_box"]:
            x1, y1, x2, y2 = s["best_box"]
            d.rectangle(
                [ox + x1 * th.width, oy + y1 * th.height, ox + x2 * th.width, oy + y2 * th.height],
                outline=(255, 210, 63) if s["kept"] else (150, 150, 150),
                width=3,
            )
    d = ImageDraw.Draw(out)
    verdict = f"{s['species']} {s['species_share']:.0%}" if s["kept"] else "dropped"
    d.text((6, TILE[1] - 21), f"#{n}  {verdict}  (P animal {s['p_animal']:.2f})", fill=(255, 255, 255))
    return out


def draw(im: Image.Image, res: FrameResult, threshold: float):
    """Frame with boxes for --show (BGR array for OpenCV)."""
    import cv2

    bgr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    for b in res.boxes:
        x1, y1, x2, y2 = (int(v * s) for v, s in zip(b.bbox, (im.width, im.height, im.width, im.height)))
        colour = (63, 210, 255) if b.p_animal >= threshold else (128, 128, 128)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), colour, 2)
        label = f"{b.species} {b.species_prob:.0%}" if b.p_animal >= threshold else f"bg? {b.p_animal:.0%}"
        cv2.putText(bgr, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    scale = 960 / max(bgr.shape[:2])
    return cv2.resize(bgr, None, fx=scale, fy=scale) if scale < 1 else bgr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="image, folder of images, video file, or webcam index")
    ap.add_argument("--config", default="configs/phase6.yaml")
    ap.add_argument("--variant", choices=["int8", "fp32"])
    ap.add_argument("--threads", type=int)
    ap.add_argument("--keep-threshold", type=float)
    ap.add_argument("--out")
    ap.add_argument("--show", action="store_true", help="live window with boxes (q to quit)")
    ap.add_argument("--max-seconds", type=float, help="stop a video or webcam after this long")
    ap.add_argument("--sheet", action="store_true", help="also write sheet.jpg, one tile per event")
    args = ap.parse_args()

    cfg = load_config(args.config)
    lc, mc = cfg["live"], cfg["models"]
    variant = args.variant or lc["variant"]
    threads = args.threads or lc["threads"]
    threshold = args.keep_threshold if args.keep_threshold is not None else lc["keep_threshold"]
    mdir = Path(mc["dir"])
    pipe = Pipeline(
        mdir / mc["detector"].format(variant=variant),
        mdir / mc["species"],
        mc["head"],
        threads=threads,
        min_conf=lc["min_conf"],
        max_boxes=lc["max_boxes"],
    )

    is_cam = args.source.isdigit()
    src_path = Path(args.source)
    if is_cam or src_path.suffix.lower() in VIDEO_EXT:
        frames = video(int(args.source) if is_cam else args.source, lc["video_fps"], args.max_seconds)
        grouper: EventGrouper | AnimalSpans = AnimalSpans(threshold, lc["video_gap"])
        name = f"webcam{args.source}" if is_cam else src_path.stem
    elif src_path.exists():
        frames = photos(src_path)
        grouper = EventGrouper(lc["gap"])
        name = src_path.stem
    else:
        raise SystemExit(f"no such file or folder: {args.source}")

    out = Path(args.out or f"outputs/phase6/live/{name}")
    (out / "crops").mkdir(parents=True, exist_ok=True)
    print(f"{variant} detector, {mc['species']}, {threads} threads, keep at P(animal) >= {threshold}; writing {out}")

    stats = {"frames": 0, "frame_bytes": 0, "events": 0, "kept": 0, "crop_bytes": 0, "meta_bytes": 0}
    times: dict[str, list[float]] = {"detector": [], "species": []}
    images: dict[str, Image.Image] = {}  # frames of the open event, to cut its crop when it closes
    tiles: list[Image.Image] = []

    def close(ev: Event | None, fe, fv) -> None:
        if ev is None:
            return
        s = ev.summary(threshold)
        stats["events"] += 1
        if s["kept"]:
            data, _ = crop_bytes(images[s["best_frame"]], s["best_box"], lc["crop"])
            fname = f"{stats['events']:04d}_{s['species']}.jpg"
            (out / "crops" / fname).write_bytes(data)
            s["crop"] = fname
            stats["kept"] += 1
            stats["crop_bytes"] += len(data)
        line = json.dumps(s)
        stats["meta_bytes"] += len(line) + 1
        fe.write(line + "\n")
        fe.flush()
        what = f"{s['species']} ({s['species_share']:.0%})" if s["kept"] else "dropped (no animal)"
        print(
            f"event {stats['events']:>4}: {s['first']}  {s['frames']} frames  P(animal) {s['p_animal']:.2f}  {what}",
            flush=True,
        )
        if args.sheet:
            shown = images.get(s["best_frame"]) if s["best_frame"] else None
            tiles.append(tile(shown if shown is not None else images.get(s["first"]), s, stats["events"]))
        for n in ev.names:
            images.pop(n, None)

    started = time.perf_counter()
    with (
        open(out / "events.jsonl", "w", encoding="utf-8") as fe,
        open(out / "frames.jsonl", "w", encoding="utf-8") as fv,
    ):
        try:
            for fname, t, im, nbytes in frames:
                res = pipe(im, t)
                stats["frames"] += 1
                stats["frame_bytes"] += nbytes
                for k, v in res.ms.items():
                    times[k].append(v)
                fv.write(
                    json.dumps(
                        {
                            "frame": fname,
                            "t": round(t, 2),
                            "ms": {k: round(v, 1) for k, v in res.ms.items()},
                            "boxes": [b.__dict__ for b in res.boxes],
                        }
                    )
                    + "\n"
                )
                done = grouper.add(res, fname)
                if isinstance(grouper, EventGrouper) or res.p_animal >= threshold:
                    images[fname] = im
                close(done, fe, fv)
                if args.show:
                    import cv2

                    cv2.imshow("ghats live (q to quit)", draw(im, res, threshold))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        except KeyboardInterrupt:
            print("stopped")
        close(grouper.flush(), fe, fv)

    wall = time.perf_counter() - started
    if args.sheet and tiles:
        cols = min(4, len(tiles))
        sheet = Image.new("RGB", (cols * TILE[0], -(-len(tiles) // cols) * TILE[1]), "black")
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % cols) * TILE[0], (i // cols) * TILE[1]))
        sheet.save(out / "sheet.jpg", quality=85)
    sent = stats["crop_bytes"] + stats["meta_bytes"]
    summary = {
        "source": args.source,
        "variant": variant,
        "threads": threads,
        "keep_threshold": threshold,
        **stats,
        "seconds": round(wall, 1),
        "ms_per_frame": {k: round(float(np.median(v)), 1) for k, v in times.items() if v},
        "ms_per_frame_p95": {k: round(float(np.percentile(v, 95)), 1) for k, v in times.items() if v},
        "sent_vs_all_frames": round(sent / stats["frame_bytes"], 4) if stats["frame_bytes"] else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.show:
        import cv2

        cv2.destroyAllWindows()
    mb = stats["frame_bytes"] / 1e6
    print(
        f"\n{stats['frames']} frames, {stats['events']} events, {stats['kept']} kept. "
        f"Median ms/frame: {summary['ms_per_frame']}. "
        f"Would send {sent / 1e3:.0f} KB instead of {mb:.1f} MB of frames"
        + (f" ({summary['sent_vs_all_frames']:.1%})." if summary["sent_vs_all_frames"] is not None else ".")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
