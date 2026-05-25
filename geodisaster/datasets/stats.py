"""Per-source normalization statistics.

Sentinel-1 dB values cluster around -25, Sentinel-2 reflectance is 0–10000,
DEM elevation is meters, AlphaEarth embeddings are roughly unit-norm. Without
normalization, the loss is dominated by whichever source has the largest
magnitude. This module computes per-source (mean, std) over a training patch
list using a numerically stable Welford accumulator and persists them as YAML.

Empirical fallbacks (used when no stats file is supplied) come from the
respective product docs / community recipes; override them by running
``geodisaster compute-stats`` on your own training patches.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import yaml


# Fallbacks: rough population means / stds for each source. Per-channel where
# the source has > 1 band; otherwise a scalar pair applied to all channels.
EMPIRICAL_FALLBACKS: dict[str, tuple[float, float]] = {
    "alphaearth": (0.0, 1.0),       # already approximately unit-norm
    "sentinel1":  (-15.0, 5.0),     # VV/VH dB, std across land+water
    "sentinel2":  (1500.0, 1500.0), # SR DN; harmonized scale
    "dem":        (500.0, 500.0),   # elevation+derivatives — coarse global guess
    "worldpop":   (10.0, 50.0),     # 100 m people/cell — heavy-tailed
}


@dataclass
class _Welford:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update_array(self, arr: np.ndarray) -> None:
        a = np.asarray(arr, dtype=np.float64).ravel()
        a = a[np.isfinite(a)]
        if a.size == 0:
            return
        n_b = a.size
        mean_b = a.mean()
        m2_b = ((a - mean_b) ** 2).sum()
        if self.n == 0:
            self.n = n_b
            self.mean = float(mean_b)
            self.m2 = float(m2_b)
            return
        delta = mean_b - self.mean
        new_n = self.n + n_b
        self.mean = self.mean + delta * (n_b / new_n)
        self.m2 = self.m2 + m2_b + (delta ** 2) * (self.n * n_b / new_n)
        self.n = new_n

    @property
    def std(self) -> float:
        if self.n < 2:
            return 1.0
        return float(np.sqrt(self.m2 / (self.n - 1)))


def compute_norm_stats(
    patches: Sequence[dict],
    sources: Sequence[str],
    max_patches: int | None = 2000,
    rng_seed: int = 1234,
) -> dict[str, tuple[float, float]]:
    """Compute (mean, std) per source across patches.

    ``max_patches`` randomly sub-samples to keep this fast; pass None to use
    everything. Patches with missing source paths are silently skipped.
    """
    if max_patches is not None and len(patches) > max_patches:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(len(patches), size=max_patches, replace=False)
        patches = [patches[i] for i in idx]

    accs: dict[str, _Welford] = {s: _Welford() for s in sources}
    for rec in patches:
        for s in sources:
            path = rec.get("sources", {}).get(s)
            if path is None:
                continue
            try:
                arr = np.load(path)
            except Exception:
                continue
            accs[s].update_array(arr)

    out: dict[str, tuple[float, float]] = {}
    for s, w in accs.items():
        if w.n == 0:
            out[s] = EMPIRICAL_FALLBACKS.get(s, (0.0, 1.0))
        else:
            out[s] = (w.mean, max(w.std, 1e-6))
    return out


def save_stats(stats: dict[str, tuple[float, float]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: {"mean": float(v[0]), "std": float(v[1])} for k, v in stats.items()}
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def load_stats(path: str | Path) -> dict[str, tuple[float, float]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {k: (float(v["mean"]), float(v["std"])) for k, v in raw.items()}


def stats_with_fallbacks(
    stats_path: str | Path | None,
    sources: Iterable[str],
) -> dict[str, tuple[float, float]]:
    """Convenience: load stats if path provided, fill missing sources with
    EMPIRICAL_FALLBACKS, and return a complete normalize-dict for the dataset.
    """
    out: dict[str, tuple[float, float]] = {}
    loaded: dict[str, tuple[float, float]] = {}
    if stats_path is not None and Path(stats_path).exists():
        loaded = load_stats(stats_path)
    for s in sources:
        if s in loaded:
            out[s] = loaded[s]
        else:
            out[s] = EMPIRICAL_FALLBACKS.get(s, (0.0, 1.0))
    return out
