"""`geodisaster build-report` — render outputs/site/index.html from current results."""
from __future__ import annotations

import argparse

from ..report import build_report
from ..utils.logging import get_logger

log = get_logger("cli.report")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser("geodisaster build-report")
    p.add_argument("--out", default="outputs/site", help="output directory")
    p.add_argument("--root", default=".", help="project root")
    p.add_argument("--serve", type=int, default=None,
                   help="after building, also start `python -m http.server <PORT>` here")
    args = p.parse_args(argv)

    path = build_report(out_dir=args.out, project_root=args.root)
    log.info("report_built", path=str(path),
             size_kb=int(path.stat().st_size / 1024))
    print(f"\nReport ready: {path}")
    print(f"Open locally:  file://{path.resolve()}")
    print(f"Serve locally: cd {args.out} && python -m http.server 8000")

    if args.serve:
        import http.server
        import os
        import socketserver
        os.chdir(args.out)
        with socketserver.TCPServer(("0.0.0.0", args.serve),
                                    http.server.SimpleHTTPRequestHandler) as httpd:
            print(f"\nServing http://0.0.0.0:{args.serve} (Ctrl-C to stop)")
            httpd.serve_forever()
    return 0
