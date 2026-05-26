"""`geodisaster build-blog` — render DeepMind-style research narrative."""
from __future__ import annotations

import argparse

from ..blog import build_blog
from ..utils.logging import get_logger

log = get_logger("cli.blog")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser("geodisaster build-blog")
    p.add_argument("--out", default="outputs/site/index.html")
    args = p.parse_args(argv)
    path = build_blog(out_path=args.out)
    size_kb = int(path.stat().st_size / 1024)
    log.info("blog_built", path=str(path), size_kb=size_kb)
    print(f"\nBlog ready: {path}  ({size_kb} KB)")
    return 0
