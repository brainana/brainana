#!/usr/bin/env python3
"""Live dashboard for segmentation-CNN training runs.

Reads only the files the trainer writes to <run>/monitor/ (metrics.jsonl, status.json,
snapshots/*.png), so it can run on any machine that sees the output folder and never
touches the training process.

    # every run under a training_output folder (new runs appear automatically)
    python -m fastsurfer_nn.scripts.training_dashboard \
        --root /mnt/DataDrive3/xliu/monkey_training_groundtruth/FastSurferCNN_training/training_output

    # or explicit runs
    python -m fastsurfer_nn.scripts.training_dashboard --runs OUT_axial OUT_coronal OUT_sagittal

Then open http://localhost:8765 (VS Code forwards the port automatically when run in its
terminal). The page polls every 5 s. Binds to 127.0.0.1 unless --host is given.

    # static copy (data + snapshots embedded), e.g. to keep next to the shipped checkpoint
    python -m fastsurfer_nn.scripts.training_dashboard --runs ... --export report.html
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PAGE = Path(__file__).with_name("training_dashboard.html")


def plotly_js() -> bytes:
    import plotly

    return (Path(plotly.__file__).parent / "package_data" / "plotly.min.js").read_bytes()


class Runs:
    def __init__(self, runs: list[Path], root: Path | None):
        self.fixed = [Path(r).resolve() for r in runs]
        self.root = Path(root).resolve() if root else None

    def all(self) -> dict[str, Path]:
        dirs = list(self.fixed)
        if self.root and self.root.exists():
            dirs += sorted(p for p in self.root.iterdir() if (p / "monitor").is_dir())
        out = {}
        for d in dirs:
            if (d / "monitor").is_dir():
                out.setdefault(d.name, d)
        return out

    def get(self, name: str) -> Path | None:
        return self.all().get(name)


def read_status(run: Path) -> dict:
    try:
        return json.loads((run / "monitor" / "status.json").read_text())
    except Exception:
        return {}


def read_records(run: Path, since: int) -> tuple[list[dict], int]:
    f = run / "monitor" / "metrics.jsonl"
    if not f.exists():
        return [], 0
    with open(f, "rb") as fh:
        fh.seek(since)
        chunk = fh.read()
    end = chunk.rfind(b"\n") + 1  # only complete lines
    recs = []
    for line in chunk[:end].splitlines():
        try:
            recs.append(json.loads(line))
        except Exception:
            pass
    return recs, since + end


def snapshot_epochs(run: Path) -> list[int]:
    out = []
    for p in (run / "monitor" / "snapshots").glob("epoch_*.png"):
        m = re.match(r"epoch_(\d+)\.png$", p.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def make_handler(runs: Runs):
    page = PAGE.read_text()
    pjs = plotly_js()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body: bytes, ctype: str, code=HTTPStatus.OK, cache=False):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj):
            self._send(json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path == "/":
                return self._send(page.replace("/*__STATIC_DATA__*/null", "null").encode(),
                                  "text/html; charset=utf-8")
            if u.path == "/plotly.min.js":
                return self._send(pjs, "application/javascript", cache=True)
            if u.path == "/api/runs":
                return self._json([{"name": n, "status": read_status(p)}
                                   for n, p in runs.all().items()])
            run = runs.get(q.get("run", ""))
            if run is None:
                return self._send(b"unknown run", "text/plain", HTTPStatus.NOT_FOUND)
            if u.path == "/api/data":
                recs, off = read_records(run, int(q.get("since", 0)))
                return self._json({"records": recs, "offset": off, "status": read_status(run),
                                   "snapshots": snapshot_epochs(run)})
            if u.path == "/api/snapshot":
                f = run / "monitor" / "snapshots" / f"epoch_{int(q.get('epoch', 0)):03d}.png"
                if f.exists():
                    return self._send(f.read_bytes(), "image/png", cache=True)
                return self._send(b"no snapshot", "text/plain", HTTPStatus.NOT_FOUND)
            self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)

    return H


def export(runs: Runs, out: Path) -> None:
    data = {}
    for name, run in runs.all().items():
        recs, _ = read_records(run, 0)
        snaps = {}
        for e in snapshot_epochs(run)[-3:]:  # keep the file small: last 3 snapshots
            b = (run / "monitor" / "snapshots" / f"epoch_{e:03d}.png").read_bytes()
            snaps[e] = "data:image/png;base64," + base64.b64encode(b).decode()
        data[name] = {"records": recs, "status": read_status(run), "snapshots": snaps}
    page = PAGE.read_text()
    page = page.replace('<script src="/plotly.min.js"></script>',
                        "<script>" + plotly_js().decode() + "</script>")
    page = page.replace("/*__STATIC_DATA__*/null", json.dumps(data))
    out.write_text(page)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(data)} runs)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="*", default=[], type=Path, help="training output folders")
    p.add_argument("--root", type=Path, default=None,
                   help="folder whose sub-folders are runs (auto-discovered)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--export", type=Path, default=None, help="write a static HTML and exit")
    args = p.parse_args()
    if not args.runs and not args.root:
        p.error("give --runs and/or --root")
    runs = Runs(args.runs, args.root)
    if args.export:
        export(runs, args.export)
        return
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(runs))
    print(f"dashboard: http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}"
          f":{args.port}  ({len(runs.all())} run(s) now; Ctrl+C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
