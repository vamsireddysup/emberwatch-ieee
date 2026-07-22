"""Dependency-free local EmberWatch telemetry dashboard and JSON API."""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HTML = REPO_ROOT / "web" / "dashboard.html"
DEFAULT_LOG = REPO_ROOT / "artifacts" / "telemetry" / "receiver_log.csv"


def read_telemetry(path: Path, limit: int = 500) -> list[dict]:
    if not path.exists():
        return []
    rows: deque[dict] = deque(maxlen=max(1, min(limit, 5000)))
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for field in ("asset_temp_c", "ambient_temp_c", "delta_c", "confidence", "snr_db"):
                if field in row:
                    row[field] = float(row[field])
            for field in ("device_id", "sequence", "uptime_s", "state", "battery_mv", "flags", "rssi_dbm"):
                if field in row:
                    row[field] = int(row[field])
            rows.append(row)
    return list(rows)


def make_handler(log_path: Path, html_path: Path):
    class DashboardHandler(BaseHTTPRequestHandler):
        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self._send(200, "text/html; charset=utf-8", html_path.read_bytes())
                return
            if parsed.path == "/api/telemetry":
                query = parse_qs(parsed.query)
                try:
                    limit = int(query.get("limit", ["500"])[0])
                except ValueError:
                    limit = 500
                body = json.dumps({"rows": read_telemetry(log_path, limit)}).encode("utf-8")
                self._send(200, "application/json", body)
                return
            if parsed.path == "/health":
                self._send(200, "application/json", b'{"status":"ok"}')
                return
            self._send(404, "application/json", b'{"error":"not found"}')

        def log_message(self, format, *args):
            return

    return DashboardHandler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    args = parser.parse_args()
    if not args.html.exists():
        raise SystemExit(f"Dashboard HTML not found: {args.html}")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.log, args.html))
    print(f"EmberWatch dashboard: http://{args.host}:{args.port}")
    print(f"Telemetry log: {args.log}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
