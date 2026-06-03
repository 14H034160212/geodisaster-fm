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
    """Active-Calibration MDP for label-efficient threshold calibration.

    Two reward modes formalising the same MDP at different objective levels:
      - "pixel"         : reward = test-set pixel F1 gain (the standard signal).
      - "decision_area" : reward = decrease in mean absolute relative error of
                          the per-chip predicted flooded AREA — the
                          decision-level quantity a responder actually asks for.
                          Requires `test_probs_per_chip` and
                          `test_labels_per_chip` (per-chip, not concatenated).

    pool_probs / pool_labels : list of per-chip flat prob/label arrays (pool)
    test_probs / test_labels : concatenated flat arrays for the test set
    test_probs_per_chip / test_labels_per_chip :
        per-chip lists used by the decision-level reward (optional unless
        reward_mode != "pixel").
    chip_features            : (n_pool, feat_dim) per-chip feature matrix
    """
    pool_probs: list
    pool_labels: list
    test_probs: np.ndarray
    test_labels: np.ndarray
    chip_features: np.ndarray
    budget: int = 4
    reward_mode: str = "pixel"
    test_probs_per_chip: list | None = None
    test_labels_per_chip: list | None = None

    def __post_init__(self):
        self.n = len(self.pool_probs)
        self.feat_dim = self.chip_features.shape[1]
        self.base_f1 = f1_at_threshold(self.test_probs, self.test_labels, 0.5)
        if self.reward_mode == "decision_area":
            assert self.test_probs_per_chip is not None and self.test_labels_per_chip is not None, \
                "decision_area reward requires test_probs_per_chip + test_labels_per_chip"
            # pre-compute per-chip GT areas (in pixel count)
            self._gt_areas = np.array(
                [float((l == 1).sum()) for l in self.test_labels_per_chip], dtype=np.float64)
            self.base_decision_error = self._calibrated_decision_error_at(0.5)
        self.reset()

    def reset(self):
        self.selected: list[int] = []
        self.steps = 0
        self.prev_f1 = self.base_f1
        if self.reward_mode == "decision_area":
            self.prev_decision_error = self.base_decision_error
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

    def _calibrated_threshold(self) -> float:
        if not self.selected:
            return 0.5
        probs = np.concatenate([self.pool_probs[i] for i in self.selected])
        labels = np.concatenate([self.pool_labels[i] for i in self.selected])
        t, _ = best_threshold_f1(probs, labels)
        return t

    def _calibrated_test_f1(self) -> float:
        return f1_at_threshold(self.test_probs, self.test_labels, self._calibrated_threshold())

    def _calibrated_decision_error_at(self, threshold: float) -> float:
        """Mean absolute relative AREA error across test chips at given threshold."""
        errs = []
        for p, gt_area in zip(self.test_probs_per_chip, self._gt_areas):
            pred_area = float((p >= threshold).sum())
            denom = max(gt_area, 100.0)   # avoid div-by-zero on dry chips
            errs.append(abs(pred_area - gt_area) / denom)
        return float(np.mean(errs))

    def _calibrated_decision_error(self) -> float:
        return self._calibrated_decision_error_at(self._calibrated_threshold())

    def step(self, action: int):
        # mask invalid (already-selected) actions by no-op penalty
        if action in self.selected:
            return self._obs(), -0.05, self.steps >= self.budget, {}
        self.selected.append(action)
        self.steps += 1
        if self.reward_mode == "decision_area":
            err = self._calibrated_decision_error()
            reward = self.prev_decision_error - err     # decrease in error
            self.prev_decision_error = err
            done = self.steps >= self.budget
            return self._obs(), reward, done, {"decision_error": err}
        f1 = self._calibrated_test_f1()
        done = self.steps >= self.budget
        if self.reward_mode == "terminal_pixel":
            reward = (f1 - self.base_f1) if done else 0.0   # episode-level signal only
        else:
            reward = f1 - self.prev_f1                       # incremental F1 gain
        self.prev_f1 = f1
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
              gamma=0.99, gae_lambda=0.95, clip=0.2, lr=3e-3, seed=0, log_every=25,
              ent_start=0.10, ent_end=0.01):
    """Train the chip-selection policy with PPO + GAE-λ + entropy annealing.

    Improvements over the original (motivated by the LOEO negative result):
      * GAE-λ for credit assignment — drastically lower-variance advantage
        estimates than raw discounted returns (which were drowning in noise
        given an episode-level F1 gain of ~0.005 with step-level σ ~0.05).
      * Linear entropy schedule (ent_start → ent_end) to keep exploration
        alive in the early phase where the policy is otherwise prone to
        collapsing onto a near-random uniform strategy.
      * Works seamlessly with the new ``reward_mode='terminal_pixel'`` mode
        of ``ChipCalibEnv`` (recommended): rewards are 0 every step except
        the terminal step, where the reward equals the full F1 gain. This
        matches the actual optimisation objective and removes the noisy
        intermediate signal.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    policy = ChipPolicy(feat_dim)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    history = []

    for upd in range(n_updates):
        ent_coef = ent_start + (ent_end - ent_start) * (upd / max(1, n_updates - 1))
        batch_obs, batch_act, batch_logp, batch_adv, batch_ret, batch_val, batch_valid = [], [], [], [], [], [], []
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
            ep_returns.append(sum(t[4] for t in traj))
            # GAE-λ advantage and return-to-go (bootstrap V(s_T)=0 at terminal)
            T = len(traj)
            gae = 0.0
            advs = [0.0] * T
            rets = [0.0] * T
            next_val = 0.0
            for tstep in range(T - 1, -1, -1):
                _, _, _, _, r, val = traj[tstep]
                v = float(val.item())
                delta = r + gamma * next_val - v
                gae = delta + gamma * gae_lambda * gae
                advs[tstep] = gae
                rets[tstep] = gae + v
                next_val = v
            for (ot, valid, a, logp, r, val), A, Rt in zip(traj, advs, rets):
                batch_obs.append(ot); batch_valid.append(valid)
                batch_act.append(a); batch_logp.append(logp)
                batch_adv.append(A); batch_ret.append(Rt); batch_val.append(val)

        adv_t = torch.tensor(batch_adv, dtype=torch.float32)
        rets_t = torch.tensor(batch_ret, dtype=torch.float32)
        # global advantage standardisation
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
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
            s1 = ratio * adv_t
            s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t
            pg_loss = -torch.min(s1, s2).mean()
            v_loss = F.mse_loss(new_val, rets_t)
            loss = pg_loss + 0.5 * v_loss - ent_coef * ent
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()

        mean_ret = float(np.mean(ep_returns))
        history.append(mean_ret)
        if upd % log_every == 0 or upd == n_updates - 1:
            print(f"  upd {upd:3d}  mean episode F1-gain return = {mean_ret:+.4f}  ent_coef={ent_coef:.3f}")

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
