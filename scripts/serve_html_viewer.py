"""Serve the results viewer and dataset images, without exposing other files.

Usage: python scripts/serve_html_viewer.py --host 0.0.0.0 --port 8004
"""

import argparse
import gzip
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote, urlsplit


PROJECT = Path(__file__).resolve().parent.parent
VIEWER = PROJECT / "outputs/html_viewer.html"
IMAGES = (PROJECT / "dataset/images").resolve()
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".avif", ".bmp"}


@lru_cache(maxsize=1)
def compressed_viewer(mtime_ns, size):
    # Stat arguments invalidate the cache when the generated viewer changes.
    return gzip.compress(VIEWER.read_bytes(), compresslevel=6, mtime=0)


def accepts_gzip(header):
    qualities = {}
    for entry in header.lower().split(","):
        coding, *parameters = entry.strip().split(";")
        quality = 1.0
        for parameter in parameters:
            key, separator, value = parameter.strip().partition("=")
            if key == "q" and separator:
                try:
                    quality = float(value)
                except ValueError:
                    quality = 0.0
        qualities[coding] = quality
    return 0 < qualities.get("gzip", qualities.get("*", 0)) <= 1


class ViewerHandler(SimpleHTTPRequestHandler):
    def allowed_file(self):
        path = unquote(urlsplit(self.path).path)
        if path == "/":
            return VIEWER
        prefix = "/dataset/images/"
        if path.startswith(prefix):
            try:
                candidate = (IMAGES / path[len(prefix):]).resolve()
                candidate.relative_to(IMAGES)
            except (ValueError, OSError):
                return None
            if candidate.suffix.lower() in IMAGE_EXTENSIONS:
                return candidate
        return None

    def send_head(self):
        try:
            target = self.allowed_file()
            if target is None or not target.is_file():
                self.send_error(404, "Not found")
                return None
            stat = target.stat()
            compressed = target == VIEWER and accepts_gzip(
                self.headers.get("Accept-Encoding", "")
            )
            if compressed:
                payload = compressed_viewer(stat.st_mtime_ns, stat.st_size)
                stream = BytesIO(payload)
                length = len(payload)
            else:
                stream = target.open("rb")
                length = stat.st_size
        except (OSError, ValueError):
            self.send_error(404, "Not found")
            return None
        self.send_response(200)
        self.send_header("Content-Type", self.guess_type(str(target)))
        self.send_header("Content-Length", str(length))
        if target == VIEWER:
            self.send_header("Vary", "Accept-Encoding")
        if compressed:
            self.send_header("Content-Encoding", "gzip")
        self.end_headers()
        return stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8004)
    args = parser.parse_args()
    if not VIEWER.is_file():
        parser.error("Viewer missing; run python outputs/build_html_viewer.py first")
    with ThreadingHTTPServer((args.host, args.port), ViewerHandler) as server:
        print(f"Viewer: http://{args.host}:{args.port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
