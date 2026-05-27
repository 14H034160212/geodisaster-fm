"""Regenerate derived figures from their JSON/CSV sources, only when stale.

Called by the watcher before each blog rebuild so the website always
reflects the latest experiment outputs. A figure is re-rendered only if
its source file is newer than the PNG (cheap no-op otherwise).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _stale(src: str, png: str) -> bool:
    s, p = Path(src), Path(png)
    if not s.exists():
        return False
    if not p.exists():
        return True
    return s.stat().st_mtime > p.stat().st_mtime


def main() -> int:
    refreshed = []

    # Fig 10 — active adaptation (Layer 3)
    adapt = "outputs/active_adapt/adapt_Pakistan.json"
    fig10 = "outputs/figures/fig10_active_adapt.png"
    if _stale(adapt, fig10):
        from render_fig10_active_adapt import render
        render(adapt, fig10)
        refreshed.append("fig10")

    # (fig8 / fig9 are one-shot; re-render only if their sources change —
    #  handled by their own scripts. Add here if they become live.)

    if refreshed:
        print(f"[refresh] re-rendered: {', '.join(refreshed)}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
