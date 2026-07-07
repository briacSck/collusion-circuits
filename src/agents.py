"""Pricing agents: tabular Q-learning pairs (Calvano et al. 2020) and
designed anchor policies, both distilled into auditable MLPs.

Why this architecture (design doc §5.2 + risk section): collusion emerges
reliably with tabular Q-learning on the memory-one state (both firms' last
prices); NN training from scratch is unstable. So we train tabular
Q-learners, then DISTILL each firm's greedy policy into a small MLP by
behavior cloning (paper #1's technique). The distilled network is what the
regulator audits — realistic (deployed pricing systems are function
approximators) and it guarantees the audit target exists whatever the RL
training does.

Anchors (ground truth by construction): grim trigger, tit-for-tat, myopic
best response, random — distilled into the same MLP architecture.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from market import BertrandMarket


# ------------------------------------------------------------- Q-learning --
@dataclass
class QPair:
    q1: np.ndarray  # (n_states, n_prices)
    q2: np.ndarray
    avg_price_last: float
    collusion_index: float


def state_index(i1: int, i2: int, n: int) -> int:
    """Memory-one state: (my last price idx, rival last price idx)."""
    return i1 * n + i2


def train_q_pair(mkt: BertrandMarket, seed: int,
                 n_steps: int = 1_500_000, alpha: float = 0.15,
                 beta_decay: float = 4e-6, delta: float = 0.95,
                 last_window: int = 20_000) -> QPair:
    """Two independent Q-learners in self-play (Calvano et al. parameters).

    Exploration: epsilon_t = exp(-beta_decay * t). Returns final Q tables and
    the collusion index over the last `last_window` periods.
    """
    rng = np.random.default_rng(seed)
    grid = mkt.price_grid()
    n = len(grid)
    profit = np.empty((n, n, 2))
    for i in range(n):
        for j in range(n):
            profit[i, j] = mkt.profits(np.array([grid[i], grid[j]]))

    q1 = rng.uniform(0, 0.5, size=(n * n, n))
    q2 = rng.uniform(0, 0.5, size=(n * n, n))
    a1, a2 = rng.integers(n), rng.integers(n)
    prices_sum, prices_cnt = 0.0, 0

    for t in range(n_steps):
        eps = np.exp(-beta_decay * t)
        s1 = state_index(a1, a2, n)
        s2 = state_index(a2, a1, n)
        na1 = rng.integers(n) if rng.random() < eps else int(np.argmax(q1[s1]))
        na2 = rng.integers(n) if rng.random() < eps else int(np.argmax(q2[s2]))

        r1, r2 = profit[na1, na2]
        ns1 = state_index(na1, na2, n)
        ns2 = state_index(na2, na1, n)
        q1[s1, na1] += alpha * (r1 + delta * q1[ns1].max() - q1[s1, na1])
        q2[s2, na2] += alpha * (r2 + delta * q2[ns2].max() - q2[s2, na2])
        a1, a2 = na1, na2

        if t >= n_steps - last_window:
            prices_sum += (grid[a1] + grid[a2]) / 2
            prices_cnt += 1

    avg_p = prices_sum / prices_cnt
    return QPair(q1=q1, q2=q2, avg_price_last=avg_p,
                 collusion_index=mkt.collusion_index(avg_p))


# ------------------------------------------------------------- anchors ----
def anchor_policy(name: str, mkt: BertrandMarket) -> np.ndarray:
    """Deterministic target policy over the memory-one state grid.

    Returns (n_states, n_prices) one-hot-ish action probabilities.
    Convention: state = (own last idx i, rival last idx j).
    """
    grid = mkt.price_grid()
    n = len(grid)
    br = mkt.static_br_on_grid()
    pn_idx = int(np.argmin(np.abs(grid - mkt.nash_price)))
    pm_idx = int(np.argmin(np.abs(grid - mkt.monopoly_price)))

    pol = np.zeros((n * n, n))
    for i in range(n):
        for j in range(n):
            s = state_index(i, j, n)
            if name == "myopic_br":
                a = br[j]
            elif name == "grim_trigger":
                # cooperate at monopoly unless rival ever priced below it
                a = pm_idx if j >= pm_idx else pn_idx
            elif name == "tit_for_tat":
                a = j  # match rival's last price
            elif name == "random":
                pol[s] = 1.0 / n
                continue
            else:
                raise ValueError(name)
            pol[s, a] = 1.0
    return pol


ANCHORS = ("grim_trigger", "tit_for_tat", "myopic_br", "random")


# ----------------------------------------------------------- distillation --
class PricingMLP(nn.Module):
    """Auditable policy network: (own last price, rival last price),
    normalized + interaction features -> logits over the price grid."""

    def __init__(self, n_prices: int, hidden: int = 32, n_layers: int = 2):
        super().__init__()
        layers, d = [], 4
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers.append(nn.Linear(d, n_prices))
        self.net = nn.Sequential(*layers)

    def forward(self, feats):
        return self.net(feats)

    def hidden_activations(self, feats):
        acts, h = [], feats
        for layer in self.net:
            h = layer(h)
            if isinstance(layer, nn.ReLU):
                acts.append(h.detach())
        return acts


def state_features(n_prices: int) -> torch.Tensor:
    """(n_states, 4) features for every memory-one state."""
    idx = np.arange(n_prices * n_prices)
    own = (idx // n_prices) / (n_prices - 1)
    rival = (idx % n_prices) / (n_prices - 1)
    feats = np.stack([own, rival, own * rival, rival**2], axis=-1)
    return torch.tensor(feats, dtype=torch.float32)


def q_to_policy(q: np.ndarray) -> np.ndarray:
    """Greedy policy from a Q table, as one-hot action probabilities."""
    pol = np.zeros_like(q)
    pol[np.arange(len(q)), q.argmax(axis=1)] = 1.0
    return pol


def distill(policy: np.ndarray, n_prices: int, n_epochs: int = 8000,
            lr: float = 2e-3, seed: int = 0, hidden: int = 64,
            weights: np.ndarray | None = None) -> PricingMLP:
    """Behavior-clone a tabular policy into a PricingMLP.

    `weights` (n_states,) optionally emphasizes on-path states (greedy
    self-play visitation): off-path Q-argmax entries are noise a deployed
    system would never express, so uniform cloning wastes capacity on them.
    """
    torch.manual_seed(seed)
    feats = state_features(n_prices)
    target = torch.tensor(policy, dtype=torch.float32)
    w = (torch.ones(len(policy)) if weights is None
         else torch.tensor(weights, dtype=torch.float32))
    w = w / w.sum()
    net = PricingMLP(n_prices, hidden=hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    logsoft = nn.LogSoftmax(dim=-1)
    for _ in range(n_epochs):
        opt.zero_grad()
        loss = -((target * logsoft(net(feats))).sum(dim=-1) * w).sum()
        loss.backward()
        opt.step()
    return net


def greedy_selfplay_visitation(q1: np.ndarray, q2: np.ndarray, n: int,
                               n_periods: int = 5000,
                               floor: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """State-visitation frequencies of both firms under greedy self-play,
    floored so every state keeps a little cloning weight."""
    v1 = np.zeros(n * n)
    v2 = np.zeros(n * n)
    a1 = int(q1[state_index(0, 0, n)].argmax())
    a2 = int(q2[state_index(0, 0, n)].argmax())
    for _ in range(n_periods):
        s1, s2 = state_index(a1, a2, n), state_index(a2, a1, n)
        v1[s1] += 1
        v2[s2] += 1
        a1, a2 = int(q1[s1].argmax()), int(q2[s2].argmax())
    v1 = v1 / v1.sum() + floor / (n * n)
    v2 = v2 / v2.sum() + floor / (n * n)
    return v1, v2


def mlp_greedy_policy(net: PricingMLP, n_prices: int) -> np.ndarray:
    with torch.no_grad():
        logits = net(state_features(n_prices))
    return logits.argmax(dim=-1).numpy()


if __name__ == "__main__":
    mkt = BertrandMarket()
    print("training one Q pair (short run for smoke test)...")
    pair = train_q_pair(mkt, seed=0, n_steps=300_000, last_window=5_000)
    print(f"avg price {pair.avg_price_last:.3f}, "
          f"collusion index {pair.collusion_index:.3f}")
    pol = q_to_policy(pair.q1)
    net = distill(pol, mkt.n_prices, n_epochs=1500)
    agree = (mlp_greedy_policy(net, mkt.n_prices) == pol.argmax(1)).mean()
    print(f"distillation greedy-action agreement: {agree:.3f}")
