"""Structured Decision Inference (SDI) — the headline method.

Motivation (mirrors the causal-graph lesson of Xu et al., Nat. Commun. 2022,
but applied at the DECISION level): a responder does not need a pixel map, they
need to know *which buildings / roads are affected*. The naive way — threshold
the per-pixel flood probability and flag a building if enough of its footprint
is "wet" — inherits the per-pixel noise and the cross-region mis-calibration we
documented. But flooding is not independent across structures: it is spatially
contiguous and follows terrain. We therefore infer the joint affected-state of
all buildings with a Markov-random-field over an infrastructure graph, combining
(i) per-building flood evidence, (ii) spatial smoothness between neighbours, and
(iii) an optional terrain prior. MAP inference (ICM) yields calibrated,
denoised decision answers that beat per-building thresholding on F1 vs ground
truth — without any extra labels.

Pure NumPy + scipy (KD-tree); deterministic; sub-second for thousands of nodes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SDIConfig:
    radius_m: float = 75.0      # connect buildings within this distance
    lambda_smooth: float = 1.0  # weight of the smoothness / attraction term
    sigma_m: float = 50.0       # edge-weight length scale (exp decay)
    max_neighbours: int = 12    # cap node degree for speed
    terrain_weight: float = 0.0 # weight of the optional terrain prior on "affected"
    max_iters: int = 50
    mode: str = "potts"         # "potts" (symmetric) or "attractive" (one-sided)
    init_thresh: float = 0.5    # initial labelling threshold on the evidence


def _clip_log(p, eps=1e-4):
    return -np.log(np.clip(p, eps, 1 - eps))


def build_graph(centroids_m: np.ndarray, cfg: SDIConfig):
    """Radius graph on building centroids (metric coords). Returns edge list +
    weights. Falls back to empty graph for <2 nodes."""
    from scipy.spatial import cKDTree
    n = len(centroids_m)
    if n < 2:
        return np.empty((0, 2), int), np.empty(0)
    tree = cKDTree(centroids_m)
    pairs = tree.query_pairs(r=cfg.radius_m, output_type="ndarray")
    if len(pairs) == 0:
        return np.empty((0, 2), int), np.empty(0)
    d = np.linalg.norm(centroids_m[pairs[:, 0]] - centroids_m[pairs[:, 1]], axis=1)
    w = np.exp(-(d ** 2) / (2 * cfg.sigma_m ** 2))
    # cap degree: keep strongest edges per node
    if cfg.max_neighbours and n > cfg.max_neighbours:
        from collections import defaultdict
        deg = defaultdict(int)
        order = np.argsort(-w)
        keep = []
        for k in order:
            a, b = pairs[k]
            if deg[a] < cfg.max_neighbours and deg[b] < cfg.max_neighbours:
                keep.append(k); deg[a] += 1; deg[b] += 1
        keep = np.array(sorted(keep))
        pairs, w = pairs[keep], w[keep]
    return pairs, w


def infer_affected(prob: np.ndarray, centroids_m: np.ndarray,
                   cfg: SDIConfig | None = None,
                   terrain_prior: np.ndarray | None = None) -> np.ndarray:
    """MAP inference of per-building affected state via ICM on an MRF.

    prob          : (n,) per-building flood evidence (mean predicted prob).
    centroids_m   : (n, 2) building centroids in a metric CRS.
    terrain_prior : optional (n,) in [0,1], higher => more flood-prone
                    (e.g. 1 - normalised HAND/elevation). Added to the unary.
    Returns boolean (n,) affected.
    """
    cfg = cfg or SDIConfig()
    n = len(prob)
    if n == 0:
        return np.zeros(0, bool)
    prob = np.asarray(prob, float)
    u1 = _clip_log(prob)        # cost of labelling "affected"
    u0 = _clip_log(1 - prob)    # cost of labelling "not affected"
    if terrain_prior is not None and cfg.terrain_weight:
        tp = np.clip(np.asarray(terrain_prior, float), 0, 1)
        u1 = u1 - cfg.terrain_weight * tp      # low terrain -> cheaper "affected"
        u0 = u0 - cfg.terrain_weight * (1 - tp)

    pairs, w = build_graph(centroids_m, cfg)
    # adjacency lists
    adj = [[] for _ in range(n)]
    for (a, b), wt in zip(pairs, w):
        adj[a].append((b, wt)); adj[b].append((a, wt))

    x = (prob > cfg.init_thresh).astype(np.int8)   # init from (tuned) threshold
    for _ in range(cfg.max_iters):
        changed = 0
        for i in range(n):
            if cfg.mode == "attractive":
                # one-sided: damaged neighbours make "affected" cheaper; an
                # isolated building is NOT pushed to "not-affected" (keeps recall)
                s_dmg = sum(wt for (j, wt) in adj[i] if x[j] == 1)
                e1 = u1[i] - cfg.lambda_smooth * s_dmg
                e0 = u0[i]
            else:  # symmetric Potts
                s1 = sum(wt for (j, wt) in adj[i] if x[j] != 1)
                s0 = sum(wt for (j, wt) in adj[i] if x[j] != 0)
                e1 = u1[i] + cfg.lambda_smooth * s1
                e0 = u0[i] + cfg.lambda_smooth * s0
            xi = 1 if e1 < e0 else 0
            if xi != x[i]:
                x[i] = xi; changed += 1
        if changed == 0:
            break
    return x.astype(bool)


# --------------------------------------------------------------------------- #
# Baselines for the decision-level comparison
# --------------------------------------------------------------------------- #
def baseline_raw_threshold(flood_frac_hard: np.ndarray, thresh: float = 0.2) -> np.ndarray:
    """B1: flag affected if >= `thresh` of footprint is wet under the 0.5 mask."""
    return np.asarray(flood_frac_hard) >= thresh


def baseline_any_intersection(flood_frac_hard: np.ndarray) -> np.ndarray:
    """B2 (naive over-predict): affected if footprint touches any flood pixel."""
    return np.asarray(flood_frac_hard) > 0.0


def baseline_prob_threshold(prob: np.ndarray, thresh: float) -> np.ndarray:
    """B3 (calibrated-probability ablation): affected if mean prob >= tuned thresh.
    Isolates whether STRUCTURE helps beyond just using probabilities."""
    return np.asarray(prob) >= thresh


def prf(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred = np.asarray(pred, bool); gt = np.asarray(gt, bool)
    tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
