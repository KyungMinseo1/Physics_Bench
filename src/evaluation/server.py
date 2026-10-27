"""Loopback-only reference editor; writes only per-problem sidecars."""
from __future__ import annotations
import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit
from .references import load_annotations, read_reference, save_reference
from .judge import image_paths


def make_handler(annotations_dir: Path, references: Path):
    annotations = load_annotations(annotations_dir)
    token = secrets.token_urlsafe(32)
    images = {}
    for pid, a in annotations.items():
        for index, path in enumerate(image_paths(a, annotations_dir.parent)):
            images[f'/images/{pid}/{index}'] = path

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, payload, content_type='application/json; charset=utf-8'):
            if not isinstance(payload, bytes):
                payload = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            path = unquote(urlsplit(self.path).path)
            if path == '/':
                html = Path(__file__).with_name('editor.html').read_text().replace('__TOKEN__', token)
                self.send(200, html.encode(), 'text/html; charset=utf-8')
            elif path == '/api/problems':
                self.send(200, [{'id': pid, 'text': a['question']['text'], 'objective': bool(a['question'].get('choices')),
                                 'status': read_reference(references, a)['review_status']} for pid, a in annotations.items()])
            elif path.startswith('/api/problems/') and path.rsplit('/', 1)[-1] in annotations:
                pid = path.rsplit('/', 1)[-1]; a = annotations[pid]
                self.send(200, {'annotation': a, 'reference': read_reference(references, a),
                                'images': [f'/images/{pid}/{i}' for i, _ in enumerate(a['question'].get('images', []))]})
            elif path in images:
                import mimetypes
                self.send(200, images[path].read_bytes(), mimetypes.guess_type(images[path].name)[0] or 'application/octet-stream')
            else:
                self.send(404, {'error': 'Not found'})

        def do_POST(self):
            if self.headers.get('X-Editor-Token') != token:
                self.send(403, {'error': 'Invalid editor token'}); return
            path = unquote(urlsplit(self.path).path)
            pid = path.rsplit('/', 1)[-1]
            if path != '/api/problems/' + pid or pid not in annotations:
                self.send(404, {'error': 'Not found'}); return
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 1_000_000:
                    raise ValueError('Invalid payload size')
                value = json.loads(self.rfile.read(length))
                save_reference(references, value, annotations[pid])
                self.send(200, {'saved': pid})
            except (ValueError, TypeError, KeyError) as exc:
                self.send(400, {'error': str(exc)})
    return Handler


def serve(annotations: Path, references: Path, port: int):
    with ThreadingHTTPServer(('127.0.0.1', port), make_handler(annotations, references)) as server:
        print(f'Annotation: {annotations.resolve()}', flush=True)
        print(f'기준 입력: http://127.0.0.1:{port}/ (VS Code 포트 전달 가능)', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
