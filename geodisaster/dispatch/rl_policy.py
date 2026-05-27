"""Layer 3 — PPO chip-selection policy for label-efficient threshold calibration.

Motivation. Result 2 showed the foundation/perception models have good
ranking quality (AUPRC) but lose F1 because the 0.5 decision threshold is
mis-calibrated on unseen regions. The cheapest possible adaptation is
therefore *threshold recalibration* from a few in-region labels — no
gradient fine-tuning required. The open question is *which* chips to label
so the recalibrated threshold generalises best to the rest of the region.

We formulate this as an MDP and train a PPO policy:
    state  : per-chip features for the unlabelled pool + budget remaining
    action : pick the next chip to label
    reward : gain in test-set F1 from the threshold calibrated on the
             chips selected so far
The environment is pure NumPy over cached probability maps, so an episode
is sub-second on CPU — thousands of PPO updates are feasible without a GPU.

This module is self-contained (compact actor-critic PPO, no RL deps).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Threshold calibration utilities
# --------------------------------------------------------------------------- #
def best_threshold_f1(probs: np.ndarray, labels: np.ndarray,
                      grid: np.ndarray | None = None) -> tuple[float, float]:
    """Return (threshold, F1) maximising F1 over a 1-D probability grid.

    probs / labels are flat arrays; labels in {0,1} (255 already removed)."""
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, 0.0
    pos = labels == 1
    for t in grid:
        pred = probs >= t
        tp = np.sum(pred & pos)
        fp = np.sum(pred & ~pos)
        fn = np.sum(~pred & pos)
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def f1_at_threshold(probs: np.ndarray, labels: np.ndarray, t: float) -> float:
    pred = probs >= t
    pos = labels == 1
    tp = np.sum(pred & pos); fp = np.sum(pred & ~pos); fn = np.sum(~pred & pos)
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
@dataclass
class ChipCalibEnv:
    """Chip-selection MDP for label-efficient threshold calibration.

    pool_probs / pool_labels : list of per-chip flat prob/label arrays (pool)
    test_probs / test_labels : concatenated flat arrays for the test set
    budget                   : number of chips the agent may label
    chip_features            : (n_pool, feat_dim) per-chip feature matrix
    """
    pool_probs: list
    pool_labels: list
    test_probs: np.ndarray
    test_labels: np.ndarray
    chip_features: np.ndarray
    budget: int = 4

    def __post_init__(self):
        self.n = len(self.pool_probs)
        self.feat_dim = self.chip_features.shape[1]
        # F1 of the global 0.5 threshold (zero-shot) on test — the baseline
        self.base_f1 = f1_at_threshold(self.test_probs, self.test_labels, 0.5)
        self.reset()

    def reset(self):
        self.selected: list[int] = []
        self.steps = 0
        self.prev_f1 = self.base_f1
        return self._obs()

    def _obs(self) -> np.ndarray:
        """Observation = per-chip features augmented with a 'selected' mask
        and a global budget-remaining scalar broadcast over chips."""
        mask = np.zeros((self.n, 1), dtype=np.float32)
        for i in self.selected:
            mask[i] = 1.0
        budget_left = (self.budget - self.steps) / max(self.budget, 1)
        budget_col = np.full((self.n, 1), budget_left, dtype=np.float32)
        return np.concatenate([self.chip_features, mask, budget_col], axis=1)

    def _calibrated_test_f1(self) -> float:
        if not self.selected:
            return self.base_f1
        probs = np.concatenate([self.pool_probs[i] for i in self.selected])
        labels = np.concatenate([self.pool_labels[i] for i in self.selected])
        t, _ = best_threshold_f1(probs, labels)
        return f1_at_threshold(self.test_probs, self.test_labels, t)

    def step(self, action: int):
        # mask invalid (already-selected) actions by no-op penalty
        if action in self.selected:
            return self._obs(), -0.05, self.steps >= self.budget, {}
        self.selected.append(action)
        self.steps += 1
        f1 = self._calibrated_test_f1()
        reward = f1 - self.prev_f1      # incremental F1 gain
        self.prev_f1 = f1
        done = self.steps >= self.budget
        return self._obs(), reward, done, {"f1": f1}

    def valid_actions(self) -> list[int]:
        return [i for i in range(self.n) if i not in self.selected]


# --------------------------------------------------------------------------- #
# Compact actor-critic PPO
# --------------------------------------------------------------------------- #
class ChipPolicy(nn.Module):
    """Per-chip scorer + value head. Permutation-equivariant over chips."""

    def __init__(self, feat_dim: int, hidden: int = 64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, 1)   # score per chip
        self.critic = nn.Linear(hidden, 1)  # value per chip → mean-pool

    def forward(self, obs: torch.Tensor):
        # obs: (n_chips, feat_dim)
        h = self.enc(obs)
        logits = self.actor(h).squeeze(-1)       # (n_chips,)
        value = self.critic(h).squeeze(-1).mean()  # scalar state value
        return logits, value


def masked_categorical(logits: torch.Tensor, valid_mask: torch.Tensor):
    neg = torch.full_like(logits, -1e9)
    masked = torch.where(valid_mask, logits, neg)
    return torch.distributions.Categorical(logits=masked)


def train_ppo(make_env, feat_dim: int, n_updates=300, episodes_per_update=8,
              gamma=0.99, clip=0.2, lr=3e-3, seed=0, log_every=25):
    """Train the chip-selection policy with PPO. ``make_env`` returns a fresh
    ChipCalibEnv each call (samples a region/episode)."""
    torch.manual_seed(seed); np.random.seed(seed)
    policy = ChipPolicy(feat_dim)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    history = []

    for upd in range(n_updates):
        batch_obs, batch_act, batch_logp, batch_ret, batch_val, batch_valid = [], [], [], [], [], []
        ep_returns = []
        for _ in range(episodes_per_update):
            env = make_env()
            obs = env.reset()
            traj = []
            done = False
            while not done:
                ot = torch.tensor(obs, dtype=torch.float32)
                valid = torch.zeros(env.n, dtype=torch.bool)
                for i in env.valid_actions():
                    valid[i] = True
                with torch.no_grad():
                    logits, val = policy(ot)
                    dist = masked_categorical(logits, valid)
                    a = dist.sample()
                    logp = dist.log_prob(a)
                obs2, r, done, _ = env.step(int(a.item()))
                traj.append((ot, valid, a, logp, r, val))
                obs = obs2
            # returns (discounted)
            R = 0.0
            rets = []
            for (_, _, _, _, r, _) in reversed(traj):
                R = r + gamma * R
                rets.insert(0, R)
            ep_returns.append(sum(t[4] for t in traj))
            for (ot, valid, a, logp, r, val), Rt in zip(traj, rets):
                batch_obs.append(ot); batch_valid.append(valid)
                batch_act.append(a); batch_logp.append(logp)
                batch_ret.append(Rt); batch_val.append(val)

        rets_t = torch.tensor(batch_ret, dtype=torch.float32)
        vals_t = torch.stack(batch_val).detach()
        adv = rets_t - vals_t
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        old_logp = torch.stack(batch_logp).detach()

        # PPO epochs
        for _ in range(4):
            new_logps, new_vals, ents = [], [], []
            for ot, valid, a in zip(batch_obs, batch_valid, batch_act):
                logits, val = policy(ot)
                dist = masked_categorical(logits, valid)
                new_logps.append(dist.log_prob(a))
                new_vals.append(val)
                ents.append(dist.entropy())
            new_logp = torch.stack(new_logps)
            new_val = torch.stack(new_vals)
            ent = torch.stack(ents).mean()
            ratio = torch.exp(new_logp - old_logp)
            s1 = ratio * adv
            s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
            pg_loss = -torch.min(s1, s2).mean()
            v_loss = F.mse_loss(new_val, rets_t)
            loss = pg_loss + 0.5 * v_loss - 0.01 * ent
            opt.zero_grad(); loss.backward(); opt.step()

        mean_ret = float(np.mean(ep_returns))
        history.append(mean_ret)
        if upd % log_every == 0 or upd == n_updates - 1:
            print(f"  upd {upd:3d}  mean episode F1-gain return = {mean_ret:+.4f}")

    return policy, history


@torch.no_grad()
def rollout_greedy(policy: ChipPolicy, env: ChipCalibEnv) -> dict:
    """Greedy rollout of the trained policy; returns selected chips + final F1."""
    obs = env.reset()
    done = False
    while not done:
        ot = torch.tensor(obs, dtype=torch.float32)
        valid = torch.zeros(env.n, dtype=torch.bool)
        for i in env.valid_actions():
            valid[i] = True
        logits, _ = policy(ot)
        neg = torch.full_like(logits, -1e9)
        a = int(torch.where(valid, logits, neg).argmax().item())
        obs, _, done, info = env.step(a)
    return {"selected": list(env.selected), "f1": env.prev_f1,
            "base_f1": env.base_f1, "gain": env.prev_f1 - env.base_f1}
