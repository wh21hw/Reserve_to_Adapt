"""Loopback-only, allowlisted checkpoint download with resumable byte ranges."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re

parser = argparse.ArgumentParser()
parser.add_argument('--root',required=True)
parser.add_argument('--port',type=int,default=8766)
args = parser.parse_args()
root = Path(args.root).resolve()
if root != Path('/content/imp-runs/a2w-capacity-refresh-k2-70e-v1'):
    raise ValueError('Only this declared completed experiment can be served')
files = {'/'+arm+'-'+kind+'.pt':root/arm/'office31-a2w_seed3'/(kind+'.pt')
         for arm in ('fixed2','refresh') for kind in ('best','last')}
if not all(path.is_file() for path in files.values()):
    raise FileNotFoundError('Missing a declared original model')


class Handler(BaseHTTPRequestHandler):
    def send_file(self,body):
        path = files.get(self.path)
        if path is None:
            self.send_error(404);return
        size = path.stat().st_size
        start,end = 0,size-1
        header = self.headers.get('Range')
        if header:
            match = re.fullmatch(r'bytes=(\d+)-(\d*)',header)
            if match is None:
                self.send_error(416);return
            start = int(match[1])
            if match[2]:end=min(int(match[2]),size-1)
            if start>end or start>=size:
                self.send_error(416);return
        self.send_response(206 if header else 200)
        self.send_header('Content-Type','application/octet-stream')
        self.send_header('Content-Length',str(end-start+1))
        self.send_header('Accept-Ranges','bytes')
        self.send_header('Cache-Control','no-store')
        if header:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.end_headers()
        if not body:return
        try:
            with path.open('rb') as source:
                source.seek(start)
                remaining=end-start+1
                while remaining:
                    chunk=source.read(min(1024*1024,remaining))
                    if not chunk:raise IOError('Original file shortened during backup')
                    self.wfile.write(chunk)
                    remaining-=len(chunk)
        except (BrokenPipeError,ConnectionResetError):
            pass  # Client can resume; original files never modified.

    def do_GET(self):self.send_file(True)
    def do_HEAD(self):self.send_file(False)


ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
