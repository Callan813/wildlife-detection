"""Phase 4, step 3: answer search queries over HTTP, for the dashboard.

    python scripts/31_search_server.py          # http://127.0.0.1:4318

The review API (dashboard/server) forwards /api/search, /api/similar and /images/search/*
here, so the browser only talks to one server. Loads BioCLIP 2's text encoder on the CPU
(about 1.5 GB of memory; a query takes ~50 ms) and the index from 29_embed.py.

  GET /api/search?q=civet+at+night&k=24     events, best first, and how the query was read
  GET /api/similar?event=<seq_key>&k=24     events that look like this one's best crop
  GET /images/search/frame/<image_id>       a frame, shrunk to 1280 px wide
  GET /images/search/crop/<crop_id>         a detector crop

Only frames and crops in the index are served.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ghats.config import load_config
from ghats.search_service import Search, text_encoder


def handler(search: Search):
    class H(BaseHTTPRequestHandler):
        def send(self, status: int, body: bytes, ctype: str, cache: bool = False):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            if cache:
                self.send_header("Cache-Control", "max-age=86400")
            self.end_headers()
            self.wfile.write(body)

        def json(self, status: int, obj) -> None:
            self.send(status, json.dumps(obj, allow_nan=False).encode(), "application/json")

        def do_GET(self):
            u = urlparse(self.path)
            args = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                k = min(max(int(args.get("k", 24)), 1), 200)
            except ValueError:
                return self.json(400, {"error": "k must be a number"})
            try:
                if u.path == "/api/search":
                    q = args.get("q", "").strip()
                    if not q or len(q) > 200:
                        return self.json(400, {"error": "q must be 1-200 characters"})
                    return self.json(200, search.search(q, k))
                if u.path == "/api/similar":
                    return self.json(200, search.similar(args.get("event", ""), k))
                parts = u.path.strip("/").split("/")
                if len(parts) == 4 and parts[:2] == ["images", "search"] and parts[2] in ("frame", "crop"):
                    body = search.image(parts[2], parts[3])
                    if body is not None:
                        return self.send(200, body, "image/jpeg", cache=True)
                return self.json(404, {"error": "not found"})
            except KeyError:
                return self.json(404, {"error": "no such event"})
            except Exception as e:  # noqa: BLE001 - one bad request shouldn't stop the server
                print(f"error on {self.path}: {e!r}", file=sys.stderr)
                return self.json(500, {"error": "internal error"})

        def log_message(self, fmt, *a):
            pass  # quiet: errors are printed above

    return H


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase4.yaml")
    ap.add_argument("--port", type=int, default=4318)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config(args.config)
    encode = text_encoder(Path(cfg["models"]["dir"]) / cfg["models"]["search"], args.threads)
    search = Search(cfg, encode)
    print(f"{len(search.idx.events)} events indexed; search server on http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), handler(search)).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
