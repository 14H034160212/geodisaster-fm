"""Reproducibility manifest writer (proposal §11 mitigation; Methods section)."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _git_revision(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False,
        )
        return out.stdout.strip() or None
    except FileNotFoundError:
        return None


def write_manifest(
    root: str | Path,
    artefacts: dict[str, str | Path],
    out_path: str | Path,
    extra: dict | None = None,
) -> Path:
    root = Path(root).resolve()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pkg_versions: dict[str, str] = {}
    for pkg in ("torch", "pytorch_lightning", "numpy", "pandas",
                "rasterio", "geopandas", "transformers", "earthengine-api"):
        try:
            mod = __import__(pkg.replace("-", "_"))
            pkg_versions[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            pkg_versions[pkg] = "missing"

    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "python": sys.version,
        "platform": platform.platform(),
        "git_revision": _git_revision(root),
        "env_seed": os.environ.get("PYTHONHASHSEED"),
        "package_versions": pkg_versions,
        "artefacts": {
            name: {"path": str(Path(p)), "sha256": _sha256(Path(p)) if Path(p).is_file() else None}
            for name, p in artefacts.items()
        },
        "extra": extra or {},
    }
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_path
